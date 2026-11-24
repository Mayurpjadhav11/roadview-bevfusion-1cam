import random
import types
import os
from pathlib import Path
from datetime import datetime, timezone
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.parallel import DataContainer

from carla_model_loader import build_carla_model, RESNET50_WEIGHTS
from carla_dataset_config import build_dataset, CLASSES

SEED = 42
FRAMES = 120
EPOCHS = 1
STEPS = FRAMES * EPOCHS
MAX_STEPS = int(os.environ.get("CARLA_MAX_STEPS", STEPS))
assert 1 <= MAX_STEPS <= STEPS
BASE_LR = 1e-4
OFFSET_LR = 0.0  # Frozen
PED = list(CLASSES).index("pedestrian")

RUN = datetime.now(timezone.utc).strftime(
    "carla_heatmap30_%Y%m%dT%H%M%SZ"
) + ("_smoke" if MAX_STEPS < STEPS else "_full")

OUT = Path("/outputs/experiments") / RUN
CKPT = Path("/outputs/checkpoints") / RUN
OUT.mkdir(parents=True, exist_ok=False)
CKPT.mkdir(parents=True, exist_ok=False)

def unwrap(value):
    return value.data if isinstance(value, DataContainer) else value

def gpu(value, dtype=torch.float32):
    value = unwrap(value)
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.to("cuda", dtype=dtype).contiguous()

MATRICES = [
    "camera2ego", "lidar2ego", "lidar2camera",
    "lidar2image", "camera_intrinsics",
    "camera2lidar", "img_aug_matrix",
    "lidar_aug_matrix",
]

def prepare(sample):
    result = {
        "img": gpu(sample["img"]).unsqueeze(0),
        "points": [gpu(sample["points"])],
        "metas": [unwrap(sample["metas"])],
        "gt_bboxes_3d": [
            unwrap(sample["gt_bboxes_3d"]).to("cuda")
        ],
        "gt_labels_3d": [
            gpu(sample["gt_labels_3d"], torch.long)
        ],
    }
    for key in MATRICES:
        result[key] = gpu(sample[key]).unsqueeze(0)
    return result

def training_mode(model):
    model.train()
    for module in model.modules():
        if isinstance(
            module, torch.nn.modules.batchnorm._BatchNorm
        ):
            module.eval()
        if isinstance(
            module, torch.nn.modules.dropout._DropoutNd
        ):
            module.p = 0.0
        if isinstance(module, torch.nn.MultiheadAttention):
            module.dropout = 0.0

def extract_losses(outputs):
    losses = {}
    for key, value in outputs.items():
        name = key.split("/")[-1]
        if "loss" not in name:
            continue
        if isinstance(value, (list, tuple)):
            value = sum(value)
        losses[name] = value

    required = [
        "loss_heatmap",
        "layer_-1_loss_cls",
        "layer_-1_loss_bbox",
    ]
    for key in required:
        assert key in losses, (key, list(outputs.keys()))
    return losses

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False


import json
import hashlib

MANIFEST = Path(
    "/outputs/experiments/"
    "carla_classaware_120_postfilter_manifest.json"
)

assert MANIFEST.is_file(), MANIFEST
manifest_bytes = MANIFEST.read_bytes()
manifest_hash = hashlib.sha256(
    manifest_bytes
).hexdigest()
manifest = json.loads(manifest_bytes)

assert manifest["selected_frames"] == FRAMES
assert manifest["selected_scene_count"] == FRAMES
assert manifest["classes"] == list(CLASSES)

selected_indices = [
    int(i) for i in manifest["indices"]
]
selected_scenes = list(manifest["scene_prefixes"])
selected_tokens = list(manifest["tokens"])

assert len(selected_indices) == FRAMES
assert len(set(selected_indices)) == FRAMES
assert len(set(selected_scenes)) == FRAMES
assert "299" not in selected_scenes

dataset = build_dataset("train")
assert len(dataset) == 16200

for index, token, scene in zip(
    selected_indices,
    selected_tokens,
    selected_scenes,
):
    actual = str(dataset.data_infos[index]["token"])
    assert actual == token, (index, actual, token)
    assert actual.rsplit("_", 1)[0] == scene
    assert len(
        dataset.get_ann_info(index)["gt_labels_3d"]
    ) > 0

for name in CLASSES:
    assert manifest["class_frames"][name] >= 15

# Revisit all 120 frames during each epoch.
# Shuffle separately with deterministic seeds.
schedule = []

for epoch in range(EPOCHS):
    positions = list(range(FRAMES))
    if epoch > 0:
        random.Random(
            SEED + 4219 + epoch
        ).shuffle(positions)

    schedule.extend(positions)

indices = [
    selected_indices[position]
    for position in schedule
]
scene_prefixes = [
    selected_scenes[position]
    for position in schedule
]

assert len(indices) == STEPS
assert len(set(indices)) == FRAMES
assert all(
    indices.count(index) == EPOCHS
    for index in selected_indices
)

model, _ = build_carla_model(
    pretrained=True,
    device="cpu"
)

head = model.heads["object"]
assert head.num_classes == len(CLASSES)
assert head.num_decoder_layers == 1
assert head.nms_kernel_size == 3

# Reconstruct the architecture added in Step 11.8.44.
head.class_dim_offset = torch.nn.Embedding(
    head.num_classes, 3
)

# Strict loading ensures no experimental parameters
# are silently lost.
torch.nn.init.zeros_(
    head.class_dim_offset.weight
)
assert torch.count_nonzero(
    head.class_dim_offset.weight
).item() == 0
final_heatmap_conv = head.heatmap_head[-1]
assert isinstance(final_heatmap_conv, torch.nn.Conv2d)
assert final_heatmap_conv.out_channels == len(CLASSES)
assert final_heatmap_conv.bias is not None

torch.nn.init.constant_(
    final_heatmap_conv.bias, -2.19
)

assert torch.allclose(
    final_heatmap_conv.bias.detach(),
    torch.full_like(final_heatmap_conv.bias, -2.19)
)

print(
    "Heatmap output bias:",
    final_heatmap_conv.bias.detach().tolist(),
    flush=True
)

model.cuda()

captured = {}

def dimension_hook(module, inputs, output):
    captured["center"] = output["center"]
    corrections = head.class_dim_offset(
        head.query_labels
    ).transpose(1, 2)

    result = dict(output)
    result["dim"] = output["dim"] + corrections
    return result

dim_hook = head.prediction_heads[0].register_forward_hook(
    dimension_hook
)

# Pedestrian-only heatmap suppression exemption.
original_pool = F.max_pool2d
original_forward = head.forward_single
nms_hits = {"value": 0}

def pedestrian_exempt_pool(x, kernel_size, *args, **kwargs):
    result = original_pool(x, kernel_size, *args, **kwargs)

    if (
        x.ndim == 4
        and x.shape[1] == head.num_classes
        and kernel_size == 3
    ):
        nms_hits["value"] += 1
        result = result.clone()
        if False:  # Standard 3x3 NMS pilot
            result[:, PED, :, :] = x[:, PED, 1:-1, 1:-1]

    return result

def pedestrian_exempt_forward(self, features, img_features, metas):
    with mock.patch.object(
        F, "max_pool2d", new=pedestrian_exempt_pool
    ):
        result = original_forward(
            features, img_features, metas
        )

    # Extract the CENTER AFTER the TransFusion decoder
    # has added the BEV reference position.
    def find_center(value):
        if isinstance(value, dict):
            if "center" in value:
                return value["center"]
            for item in value.values():
                found = find_center(item)
                if found is not None:
                    return found

        if isinstance(value, (list, tuple)):
            for item in value:
                found = find_center(item)
                if found is not None:
                    return found

        return None

    final_center = find_center(result)

    assert final_center is not None, (
        "Final TransFusion center not found"
    )
    assert final_center.ndim == 3
    assert final_center.shape[1] == 2
    assert final_center.shape[2] == head.num_proposals

    captured["final_center"] = final_center

    return result

head.forward_single = types.MethodType(
    pedestrian_exempt_forward, head
)

# Preserve the prior experiment's class-consistent
# dimension supervision.
original_targets = head.get_targets
diagnostics = {"mismatches": 0, "positives": 0}

def consistent_targets(*args, **kwargs):
    result = list(original_targets(*args, **kwargs))

    labels = result[0].reshape(-1)
    proposal_classes = head.query_labels.reshape(-1)

    weights = result[3]
    flat = weights.clone().reshape(-1, weights.shape[-1])

    assert len(labels) == len(proposal_classes)

    positive = flat[:, 3:6].sum(1) > 0

    mismatch = (
        positive
        & (labels != proposal_classes)
        & (labels < head.num_classes)
    )

    diagnostics["mismatches"] = int(mismatch.sum().item())
    diagnostics["positives"] = int(positive.sum().item())

    flat[mismatch, 3:6] = 0.0
    result[3] = flat.reshape_as(weights)

    diagnostics["targets"] = result[2].detach()
    diagnostics["weights"] = result[3].detach()
    diagnostics["labels"] = result[0].detach()

    return tuple(result)

head.get_targets = consistent_targets



# HEATMAP30: dense BEV heatmap branch only.

HEATMAP_PREFIX = "heads.object.heatmap_head."

trainable_names = []
trainable_params = []

for name, param in model.named_parameters():
    enabled = name.startswith(HEATMAP_PREFIX)
    param.requires_grad_(enabled)

    if enabled:
        trainable_names.append(name)
        trainable_params.append(param)

assert len(trainable_names) == 5, trainable_names

assert sum(
    param.numel() for param in trainable_params
) == 154630

assert not head.class_dim_offset.weight.requires_grad

optimizer = torch.optim.AdamW(
    [{
        "params": trainable_params,
        "lr": BASE_LR,
        "weight_decay": 0.01,
    }]
)

print(
    "HEATMAP_TRAINABLE_NAMES:",
    trainable_names,
    flush=True,
)

print(
    "HEATMAP_TRAINABLE_SCALARS:",
    sum(p.numel() for p in trainable_params),
    flush=True,
)

print(
    "HEATMAP_FROZEN_DIMENSIONS: PASS",
    flush=True,
)

from scipy.optimize import linear_sum_assignment

PILOT_SOURCE = (
    "/outputs/checkpoints/"
    "carla_dimcal_train120_zero_update_1192146/"
    "calibrated_step30_no_updates.pth"
)

source_checkpoint = torch.load(
    PILOT_SOURCE, map_location="cpu"
)

assert source_checkpoint["step"] == 30
assert source_checkpoint["dimension_calibration"][
    "additional_optimizer_updates"
] == 0

model.load_state_dict(
    source_checkpoint["model"],
    strict=True
)

DIM_KEY = "heads.object.class_dim_offset.weight"

assert torch.equal(
    head.class_dim_offset.weight.detach().cpu(),
    source_checkpoint["model"][DIM_KEY]
)

print("PEDLOCAL_STRICT_LOAD: PASS", flush=True)
print("PEDLOCAL_DIMCAL_PRESERVED: PASS", flush=True)


original_assign = head.bbox_assigner.assign

pilot_assignment = {
    "positive": 0,
    "unmatched_gt": 0,
    "mismatches": 0,
    "mean_center_distance": 0.0,
}

def pilot_xy_assign(
    bboxes, gt_bboxes, gt_labels,
    cls_pred, train_cfg
):
    # Obtain the standard AssignResult structure.
    # Its assignments are then replaced completely.
    result = original_assign(
        bboxes, gt_bboxes, gt_labels,
        cls_pred, train_cfg
    )

    result.gt_inds.zero_()
    result.labels.fill_(-1)
    result.max_overlaps.zero_()

    query_classes = (
        head.query_labels.reshape(-1)
        .long().to(bboxes.device)
    )

    assert len(query_classes) == len(bboxes)

    # IoU is retained for assignment diagnostics.
    if len(gt_labels):
        iou_matrix = head.bbox_assigner.iou_calculator(
            bboxes.detach(), gt_bboxes.detach()
        )
    else:
        iou_matrix = None

    matched_distances = []
    matched_count = 0

    for cls in range(len(CLASSES)):
        qidx = torch.nonzero(
            query_classes == cls
        ).reshape(-1)

        gidx = torch.nonzero(
            gt_labels == cls
        ).reshape(-1)

        if not len(qidx) or not len(gidx):
            continue

        xy_cost = torch.cdist(
            gt_bboxes[gidx, :2].float(),
            bboxes[qidx, :2].float()
        )

        assert torch.isfinite(xy_cost).all()

        rows, cols = linear_sum_assignment(
            xy_cost.detach().cpu().numpy()
        )

        rows = torch.as_tensor(
            rows, device=bboxes.device,
            dtype=torch.long
        )
        cols = torch.as_tensor(
            cols, device=bboxes.device,
            dtype=torch.long
        )

        selected_q = qidx[cols]
        selected_gt = gidx[rows]

        result.gt_inds[selected_q] = selected_gt + 1
        result.labels[selected_q] = gt_labels[selected_gt]

        result.max_overlaps[selected_q] = (
            iou_matrix[selected_q, selected_gt]
        )

        matched_distances.extend(
            xy_cost[rows, cols].detach().cpu().tolist()
        )
        matched_count += len(rows)

    positive = result.gt_inds > 0

    assert int(positive.sum()) == matched_count

    if matched_count:
        assigned_gt = result.gt_inds[positive] - 1
        mismatch = int((
            query_classes[positive]
            != gt_labels[assigned_gt]
        ).sum().item())
    else:
        mismatch = 0

    assert mismatch == 0

    pilot_assignment["positive"] = matched_count
    pilot_assignment["unmatched_gt"] = (
        len(gt_labels) - matched_count
    )
    pilot_assignment["mismatches"] = mismatch
    pilot_assignment["mean_center_distance"] = (
        sum(matched_distances) / matched_count
        if matched_count else 0.0
    )

    return result

head.bbox_assigner.assign = pilot_xy_assign

print("PILOT_SOURCE:", PILOT_SOURCE, flush=True)
print("PILOT_CHECKPOINT_LOAD: STRICT PASS", flush=True)
print("PILOT_NMS: STANDARD", flush=True)
print("PILOT_MATCHER: CLASS_RESTRICTED_XY", flush=True)
print("PILOT_OPTIMIZER: FRESH_ADAMW", flush=True)

print("=== STEP 11.9.22.5C: PED-LOCAL30 ===", flush=True)
print("Initialization: pretrained ResNet50", flush=True)
print("Pretrained file:", RESNET50_WEIGHTS, flush=True)
print("Manifest:", MANIFEST, flush=True)
print("Manifest SHA256:", manifest_hash, flush=True)
print("Unique frames:", FRAMES, flush=True)
print("Epochs:", EPOCHS, flush=True)
print("Unique scenes:", len(set(selected_scenes)), flush=True)
print("Selected indices:", selected_indices, flush=True)
print("Selected scene prefixes:", selected_scenes, flush=True)
print("Pedestrian class:", PED, flush=True)
print("Base LR:", BASE_LR, flush=True)
print("Offset LR:", OFFSET_LR, flush=True)
print("NMS: standard 3x3 for every class", flush=True)
print("Steps:", STEPS, flush=True)
print("Updates this run:", MAX_STEPS, flush=True)

random.seed(SEED + 149)
np.random.seed(SEED + 149)
torch.manual_seed(SEED + 149)
torch.cuda.manual_seed_all(SEED + 149)

for step, index in enumerate(indices[:MAX_STEPS], start=1):
    sample = dataset[index]

    assert sample is not None, index

    data = prepare(sample)
    n_objects = len(data["gt_labels_3d"][0])

    assert n_objects > 0, (index, "empty sample")

    training_mode(model)
    optimizer.zero_grad(set_to_none=True)
    nms_hits["value"] = 0

    outputs = model(**data)
    losses = extract_losses(outputs)
    # Extra supervision for the hardest correctly
    # assigned pedestrian, measured in metres.
    pred_center = captured["final_center"].transpose(
        1, 2
    ).reshape(-1, 2)

    target_tensor = diagnostics["targets"]
    target_center = target_tensor.reshape(
        -1, target_tensor.shape[-1]
    )[:, :2]

    weights = diagnostics["weights"]
    flat_weights = weights.reshape(
        -1, weights.shape[-1]
    )

    labels = diagnostics["labels"].reshape(-1)
    query_classes = head.query_labels.reshape(-1)

    mask = (
        (labels == PED)
        & (query_classes == PED)
        & (flat_weights[:, :2].sum(1) > 0)
    )

    valid_count = int(mask.sum().item())

    if valid_count:
        spacing = torch.as_tensor(
            head.bbox_coder.voxel_size[:2],
            device=pred_center.device,
            dtype=pred_center.dtype
        ) * float(head.bbox_coder.out_size_factor)

        center_errors = torch.linalg.vector_norm(
            (pred_center[mask] - target_center[mask])
            * spacing,
            dim=1
        )

        assert torch.isfinite(center_errors).all()

        hard_center_error = center_errors.max()
        mean_center_error = center_errors.mean()

        center_aux_loss = (
            1.5 * hard_center_error
            + 0.5 * mean_center_error
        )
    else:
        # No eligible pedestrian: use normal detection losses.
        center_aux_loss = pred_center.sum() * 0.0
        hard_center_error = center_aux_loss.detach()
        mean_center_error = center_aux_loss.detach()

    if step == 1:
        print(
            "CENTER FRAME CHECK:",
            "pred_range=",
            (
                round(float(pred_center.min().item()), 3),
                round(float(pred_center.max().item()), 3),
            ),
            "target_range=",
            (
                round(float(target_center[mask].min().item()), 3) if valid_count else None,
                round(float(target_center[mask].max().item()), 3) if valid_count else None,
            ),
            "hard_center_m=",
            round(float(hard_center_error.item()), 4),
            flush=True
        )

    # The single-scene 3m limit does not apply to
    # previously unseen training scenes.
    assert torch.isfinite(center_aux_loss).item()

    assert "layer_-1_loss_bbox" in losses
    total = losses["loss_heatmap"]

    if not torch.isfinite(total).item():
        raise RuntimeError(
            f"Nonfinite loss at step {step}"
        )

    assert nms_hits["value"] > 0, (
        "Pedestrian NMS exemption not applied"
    )

    print(
        f"PILOT_ASSIGN step={step} "
        f"GT={n_objects} "
        f"positives={pilot_assignment['positive']} "
        f"unmatched={pilot_assignment['unmatched_gt']} "
        f"mismatches={pilot_assignment['mismatches']} "
        f"dim_supervised={diagnostics['positives'] - diagnostics['mismatches']} "
        f"mean_center_m={pilot_assignment['mean_center_distance']:.4f}",
        flush=True
    )

    assert pilot_assignment["mismatches"] == 0
    assert diagnostics["mismatches"] == 0
    assert (
        diagnostics["positives"]
        == pilot_assignment["positive"]
    )

    total.backward()



    assert head.class_dim_offset.weight.grad is None

    heatmap_conv = head.heatmap_head[-1]

    assert isinstance(
        heatmap_conv, torch.nn.Conv2d
    )

    heatmap_grad = heatmap_conv.weight.grad

    assert heatmap_grad is not None
    assert heatmap_grad.shape[0] == len(CLASSES)
    assert torch.isfinite(heatmap_grad).all()

    grad_norms = heatmap_grad.reshape(
        len(CLASSES), -1
    ).norm(dim=1)

    assert torch.isfinite(grad_norms).all()
    assert float(grad_norms.sum()) > 0.0

    if step == 1 or step % 10 == 0:
        print(
            "HEATMAP_GRAD step={} class_norms={}".format(
                step,
                {
                    name: round(float(norm.item()), 9)
                    for name, norm in zip(
                        CLASSES, grad_norms
                    )
                },
            ),
            flush=True,
        )

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 35.0
    )

    if not torch.isfinite(grad_norm).item():
        raise RuntimeError(
            f"Nonfinite gradient at step {step}"
        )

    optimizer.step()

    if step == 1 or step % 10 == 0:
        print(
            f"STEP {step:03d}/{STEPS} "
            f"epoch={(step-1)//FRAMES+1}/{EPOCHS} "
            f"index={index} "
            f"scene={scene_prefixes[step-1]} "
            f"objects={n_objects} ped_pos={valid_count} "
            f"total={total.item():.5f} "
            f"heatmap={losses['loss_heatmap'].item():.5f} "
            f"cls={losses['layer_-1_loss_cls'].item():.5f} "
            f"bbox={losses['layer_-1_loss_bbox'].item():.5f} "
            f"positives={diagnostics['positives']} "
            f"mismatch={diagnostics['mismatches']} "
            f"grad={float(grad_norm):.4f} "
            f"hard_center_m={hard_center_error.item():.4f} "
            f"mean_center_m={mean_center_error.item():.4f} "
            f"center_aux={center_aux_loss.item():.4f}",
            flush=True
        )

    if step in (1, 10, 30):
        path = CKPT / f"step{step}.pth"
        torch.save(
            {
                "model": {
                    k: v.detach().cpu()
                    for k, v in model.state_dict().items()
                },
                "step": step,
                "max_updates_this_run": MAX_STEPS,
                "heatmap_output_prior_bias": "DIMCAL_checkpoint_loaded",
                "training_type": "heatmap_only",
                "trainable_scope": HEATMAP_PREFIX,
                "optimized_loss": "loss_heatmap",
                "dimension_offsets_frozen": True,
                "manifest": str(MANIFEST),
                "manifest_sha256": manifest_hash,
                "unique_frames": FRAMES,
                "epochs": EPOCHS,
                "selected_unique_indices": selected_indices,
                "schedule_seed": SEED + 4219,
                "image_backbone_pretrained": True,
                "class_dim_offset_initialization": "DIMCAL_TRAIN120_ZERO",
                "sample_indices": indices[:step],
                "scene_prefixes": scene_prefixes[:step],
                "source_checkpoint": PILOT_SOURCE,
                "initial_checkpoint": str(RESNET50_WEIGHTS),
                "class_dim_offset_enabled": True,
                "pedestrian_nms_exempt": False,
                "mismatch_dim_mask_enabled": True,
                "base_lr": BASE_LR,
                "offset_lr": OFFSET_LR,
                "hard_ped_center_loss": False,
                "hard_center_weight": 0.0,
                "mean_center_weight": 0.0,
                "pedestrian_nms_exempt": False,
            },
            path
        )
        print("CHECKPOINT SAVED:", path, flush=True)

print("\nSTEP 11.9.20 COMPLETE", flush=True)
print("Checkpoint directory:", CKPT, flush=True)
print("Experiment directory:", OUT, flush=True)
print("Original BEVFusion source unchanged.", flush=True)

head.forward_single = original_forward
head.get_targets = original_targets
dim_hook.remove()
