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

PED_NMS_MODE = os.environ["CARLA_PED_NMS_EXEMPT"]
assert PED_NMS_MODE in ("0", "1")
PED_EXEMPT = PED_NMS_MODE == "1"

SEED = 42
FRAMES = 120
EPOCHS = 3
STEPS = FRAMES * EPOCHS
MAX_STEPS = int(os.environ.get("CARLA_MAX_STEPS", STEPS))
assert 1 <= MAX_STEPS <= STEPS
BASE_LR = 1e-4
OFFSET_LR = 3e-4
PED = list(CLASSES).index("pedestrian")

RUN = datetime.now(timezone.utc).strftime(
    "carla_pretrained_120_postfilter_%Y%m%dT%H%M%SZ"
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
        if PED_EXEMPT:
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

offset_params = list(head.class_dim_offset.parameters())
offset_ids = {id(p) for p in offset_params}

shared_params = [
    p for p in model.parameters()
    if id(p) not in offset_ids
]

optimizer = torch.optim.AdamW(
    [
        {
            "params": shared_params,
            "lr": BASE_LR,
            "weight_decay": 0.01,
        },
        {
            "params": offset_params,
            "lr": OFFSET_LR,
            "weight_decay": 0.0,
        },
    ]
)


# Assignment diagnostic: use the trained CLEAN360 model.
audit_checkpoint = (
    "/outputs/checkpoints/"
    "carla_pretrained_120_postfilter_20261028T144423Z_full/"
    "step360.pth"
)

state = torch.load(audit_checkpoint, map_location="cpu")
model.load_state_dict(state["model"], strict=True)

audit_matrix = torch.zeros(
    len(CLASSES), len(CLASSES), dtype=torch.long
)
audit_gt = torch.zeros(len(CLASSES), dtype=torch.long)
audit_dim_enabled = torch.zeros(
    len(CLASSES), dtype=torch.long
)
audit_frames = 0

spatial_gt = torch.zeros(len(CLASSES), dtype=torch.long)
spatial_any_2m = torch.zeros(len(CLASSES), dtype=torch.long)
spatial_same_2m = torch.zeros(len(CLASSES), dtype=torch.long)
spatial_same_5m = torch.zeros(len(CLASSES), dtype=torch.long)
spatial_no_class = torch.zeros(len(CLASSES), dtype=torch.long)


coverage_gt = torch.zeros(len(CLASSES), dtype=torch.long)
coverage_capacity = torch.zeros(len(CLASSES), dtype=torch.long)
coverage_zero_query_gt = torch.zeros(len(CLASSES), dtype=torch.long)
coverage_zero_query_frames = torch.zeros(len(CLASSES), dtype=torch.long)
coverage_query_totals = torch.zeros(len(CLASSES), dtype=torch.long)


print("ASSIGN_AUDIT_CHECKPOINT: CLEAN360", flush=True)

print("=== STEP 11.9.20: CLEAN PRETRAINED CARLA PILOT ===", flush=True)
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
print("NMS: 3x3 except pedestrian", flush=True)
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

    with torch.no_grad():
        outputs = model(**data)
        losses = extract_losses(outputs)

    # No backward or optimizer update in this diagnostic.
    labels = diagnostics["labels"].reshape(-1).long()
    weights = diagnostics["weights"].reshape(
        -1, diagnostics["weights"].shape[-1]
    )
    query = head.query_labels.reshape(-1).long()

    positive = (
        (labels >= 0)
        & (labels < len(CLASSES))
        & (weights[:, :2].sum(1) > 0)
    )

    gt = data["gt_labels_3d"][0].long()
    audit_gt += torch.bincount(
        gt.detach().cpu(),
        minlength=len(CLASSES)
    )

    pos_query = query[positive]
    pos_gt = labels[positive]

    frame_matrix = torch.bincount(
        (
            pos_query * len(CLASSES) + pos_gt
        ).detach().cpu(),
        minlength=len(CLASSES) ** 2
    ).reshape(len(CLASSES), len(CLASSES))

    audit_matrix += frame_matrix

    dimension_active = (
        weights[positive, 3:6].sum(1) > 0
    )

    audit_dim_enabled += torch.bincount(
        pos_gt[dimension_active].detach().cpu(),
        minlength=len(CLASSES)
    )

    mismatch_count = int(
        (pos_query != pos_gt).sum().item()
    )
    positive_count = int(positive.sum().item())

    # Spatial diagnostic using the decoder-predicted
    # centers, converted into LiDAR metric coordinates.
    center_grid = (
        captured["final_center"][0]
        .transpose(0, 1)
        .detach()
    )

    assert center_grid.shape == (
        head.num_proposals, 2
    )

    spacing = torch.as_tensor(
        head.bbox_coder.voxel_size[:2],
        dtype=center_grid.dtype,
        device=center_grid.device,
    ) * float(head.bbox_coder.out_size_factor)

    origin = torch.as_tensor(
        head.bbox_coder.pc_range[:2],
        dtype=center_grid.dtype,
        device=center_grid.device,
    )

    pred_xy = center_grid * spacing + origin

    gt_xy = (
        data["gt_bboxes_3d"][0]
        .gravity_center[:, :2]
        .to(pred_xy.device)
    )

    gt_classes = gt.to(pred_xy.device)
    query_classes = query.to(pred_xy.device)

    assert len(gt_xy) == len(gt_classes)
    assert len(query_classes) == len(pred_xy)
    assert torch.isfinite(pred_xy).all()
    assert torch.isfinite(gt_xy).all()

    distances = torch.cdist(
        gt_xy.float(),
        pred_xy.float()
    )

    for cls_index, name in enumerate(CLASSES):
        gt_mask = gt_classes == cls_index
        query_mask = query_classes == cls_index

        count = int(gt_mask.sum().item())

        if count == 0:
            continue

        spatial_gt[cls_index] += count

        d_any = distances[gt_mask].min(dim=1).values
        spatial_any_2m[cls_index] += int(
            (d_any <= 2.0).sum().item()
        )

        if not query_mask.any().item():
            spatial_no_class[cls_index] += count

            print(
                f"SPATIAL_FRAME step={step} "
                f"class={name} GT={count} "
                f"queries=0 any_2m="
                f"{int((d_any <= 2.0).sum().item())} "
                f"same_2m=0 same_5m=0",
                flush=True
            )
            continue

        d_same = distances[gt_mask][:, query_mask]
        nearest = d_same.min(dim=1).values

        n2 = int((nearest <= 2.0).sum().item())
        n5 = int((nearest <= 5.0).sum().item())

        spatial_same_2m[cls_index] += n2
        spatial_same_5m[cls_index] += n5

        print(
            f"SPATIAL_FRAME step={step} "
            f"class={name} GT={count} "
            f"queries={int(query_mask.sum().item())} "
            f"any_2m={int((d_any <= 2.0).sum().item())} "
            f"same_2m={n2} same_5m={n5}",
            flush=True
        )

    audit_frames += 1

    # Number of selected queries for each class.
    q_counts = torch.bincount(
        query.detach().cpu(),
        minlength=len(CLASSES)
    )

    # Number of ground-truth objects for each class.
    gt_counts = torch.bincount(
        gt.detach().cpu(),
        minlength=len(CLASSES)
    )

    assert len(q_counts) == len(CLASSES)
    assert len(gt_counts) == len(CLASSES)
    assert int(q_counts.sum()) == head.num_proposals

    # Upper bound on class-consistent one-to-one matches.
    # This ignores geometric matching quality.
    capacity = torch.minimum(q_counts, gt_counts)

    absent = (gt_counts > 0) & (q_counts == 0)

    coverage_gt += gt_counts
    coverage_capacity += capacity
    coverage_query_totals += q_counts
    coverage_zero_query_gt += gt_counts * absent.long()
    coverage_zero_query_frames += absent.long()

    print(
        f"COVER_FRAME step={step} "
        f"gt={gt_counts.tolist()} "
        f"queries={q_counts.tolist()} "
        f"max_class_matches={int(capacity.sum())}/"
        f"{int(gt_counts.sum())} "
        f"gt_without_query={int((gt_counts * absent).sum())}",
        flush=True
    )


    print(
        f"ASSIGN_FRAME step={step} "
        f"GT={len(gt)} "
        f"positives={positive_count} "
        f"mismatches={mismatch_count} "
        f"dim_enabled={int(dimension_active.sum().item())}",
        flush=True
    )

    if step == MAX_STEPS:
        print(
            "=== ASSIGNMENT CONFUSION MATRIX ===",
            flush=True
        )

        print(
            "Rows: selected query class",
            flush=True
        )
        print(
            "Columns: matched GT class",
            flush=True
        )
        print(
            "Class order:", list(CLASSES),
            flush=True
        )

        for cls, row in zip(CLASSES, audit_matrix):
            print(
                "ASSIGN_ROW",
                cls,
                row.tolist(),
                flush=True
            )

        assigned_per_gt = audit_matrix.sum(0)
        same_class = audit_matrix.diag()
        total_pos = int(audit_matrix.sum().item())
        total_same = int(same_class.sum().item())

        for i, name in enumerate(CLASSES):
            assigned = int(assigned_per_gt[i])
            same = int(same_class[i])
            enabled = int(audit_dim_enabled[i])

            print(
                f"ASSIGN_CLASS class={name} "
                f"GT={int(audit_gt[i])} "
                f"assigned={assigned} "
                f"same_class={same} "
                f"mismatch={assigned-same} "
                f"dim_supervised={enabled}",
                flush=True
            )

        print(
            f"ASSIGN_TOTAL frames={audit_frames} "
            f"positives={total_pos} "
            f"same_class={total_same} "
            f"mismatches={total_pos-total_same} "
            f"mismatch_pct="
            f"{100*(total_pos-total_same)/max(total_pos,1):.2f}",
            flush=True
        )

        print(
            "=== CLASS-CONDITIONED SPATIAL COVERAGE ===",
            flush=True
        )

        for i, name in enumerate(CLASSES):
            print(
                f"SPATIAL_CLASS class={name} "
                f"GT={int(spatial_gt[i])} "
                f"any_center_2m={int(spatial_any_2m[i])} "
                f"same_center_2m={int(spatial_same_2m[i])} "
                f"same_center_5m={int(spatial_same_5m[i])} "
                f"GT_without_class_queries="
                f"{int(spatial_no_class[i])}",
                flush=True
            )

        print("=== CLASS QUERY COVERAGE ===", flush=True)

        for i, name in enumerate(CLASSES):
            print(
                f"COVER_CLASS class={name} "
                f"GT={int(coverage_gt[i])} "
                f"selected_queries={int(coverage_query_totals[i])} "
                f"max_class_matches={int(coverage_capacity[i])} "
                f"gt_in_zero_query_frames="
                f"{int(coverage_zero_query_gt[i])} "
                f"zero_query_frames="
                f"{int(coverage_zero_query_frames[i])}",
                flush=True
            )

        print(
            f"COVER_TOTAL GT={int(coverage_gt.sum())} "
            f"max_class_matches={int(coverage_capacity.sum())} "
            f"unmatchable_by_class_count="
            f"{int((coverage_gt-coverage_capacity).sum())}",
            flush=True
        )

        print(
            "ASSIGNMENT DIAGNOSTIC COMPLETE",
            flush=True
        )
        raise SystemExit(0)

    continue
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

    total = sum(losses.values())

    if not torch.isfinite(total).item():
        raise RuntimeError(
            f"Nonfinite loss at step {step}"
        )

    assert nms_hits["value"] > 0, (
        "Pedestrian NMS exemption not applied"
    )

    total.backward()

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

    if step in (10, 60, 120, 240, 360):
        path = CKPT / f"step{step}.pth"
        torch.save(
            {
                "model": {
                    k: v.detach().cpu()
                    for k, v in model.state_dict().items()
                },
                "step": step,
                "max_updates_this_run": MAX_STEPS,
                "heatmap_output_prior_bias": -2.19,
                "training_type": "clean_pretrained",
                "manifest": str(MANIFEST),
                "manifest_sha256": manifest_hash,
                "unique_frames": FRAMES,
                "epochs": EPOCHS,
                "selected_unique_indices": selected_indices,
                "schedule_seed": SEED + 4219,
                "image_backbone_pretrained": True,
                "class_dim_offset_initialization": "zeros",
                "sample_indices": indices[:step],
                "scene_prefixes": scene_prefixes[:step],
                "source_checkpoint": None,
                "initial_checkpoint": str(RESNET50_WEIGHTS),
                "class_dim_offset_enabled": True,
                "pedestrian_nms_exempt": True,
                "mismatch_dim_mask_enabled": True,
                "base_lr": BASE_LR,
                "offset_lr": OFFSET_LR,
                "hard_ped_center_loss": False,
                "hard_center_weight": 0.0,
                "mean_center_weight": 0.0,
                "pedestrian_nms_exempt": True,
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
