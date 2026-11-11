import os
import random
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
from mmcv.parallel import DataContainer

from carla_dataset_config import build_dataset, CLASSES
from carla_model_loader import build_carla_model

SEED = 42
STEPS = 2
LR = 1e-5

SOURCE = Path(
    "/outputs/checkpoints/"
    "carla_dimcal_train120_zero_update_1192146/"
    "calibrated_step30_no_updates.pth"
)

assert SOURCE.is_file()

RUN = "carla_class_consistent_real_" + datetime.now(
    timezone.utc
).strftime("%Y%m%dT%H%M%S%fZ")

OUT = Path("/outputs/checkpoints") / RUN
OUT.mkdir(parents=True, exist_ok=False)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False

def unwrap(x):
    return x.data if isinstance(x, DataContainer) else x

def gpu(x, dtype=torch.float32):
    x = unwrap(x)
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return x.to("cuda", dtype=dtype).contiguous()

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

    for key in (
        "camera2ego", "lidar2ego", "lidar2camera",
        "lidar2image", "camera_intrinsics",
        "camera2lidar", "img_aug_matrix",
        "lidar_aug_matrix",
    ):
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

print("=== STEP 11.9.24.3 ===", flush=True)
print("FULL-MODEL TRAINING SMOKE TEST", flush=True)
print("Run:", RUN, flush=True)

dataset = build_dataset("train")
assert len(dataset) == 16200



# Controlled regression test: one nonempty-GT frame followed
# by the exact empty-GT frame that previously caused failure.
indices = [6811, 7343]
selected_scenes = ["282", "165"]

assert len(indices) == STEPS
assert len(set(indices)) == STEPS

print("TEST INDICES:", indices, flush=True)

for index, expected_count in ((6811, 13), (7343, 0)):
    sample = dataset[index]
    assert sample is not None

    labels = unwrap(sample["gt_labels_3d"])
    boxes = unwrap(sample["gt_bboxes_3d"])

    assert len(labels) == expected_count, (
        index, len(labels), expected_count
    )
    assert len(boxes) == expected_count

    print(
        "VERIFIED FRAME:",
        index,
        "GT:",
        len(labels),
        flush=True,
    )

print("Training indices:", indices, flush=True)
print(
    "Training tokens:",
    [str(dataset.data_infos[i]["token"]) for i in indices],
    flush=True
)

model, _ = build_carla_model(
    pretrained=False, device="cpu"
)

head = model.heads["object"]

assert head.num_classes == len(CLASSES)
assert head.nms_kernel_size == 3

head.class_dim_offset = torch.nn.Embedding(
    head.num_classes, 3
)

state = torch.load(SOURCE, map_location="cpu")
model.load_state_dict(state["model"], strict=True)

assert state["dimension_calibration"][
    "additional_optimizer_updates"
] == 0

del state

def dimension_hook(module, inputs, output):
    result = dict(output)
    correction = head.class_dim_offset(
        head.query_labels
    ).transpose(1, 2)

    result["dim"] = output["dim"] + correction
    return result

dim_handle = head.prediction_heads[0].register_forward_hook(
    dimension_hook
)

# Preserve the previous class-consistent dimension
# supervision, without the pedestrian NMS exemption.

# Repair the empty-GT assignment contract without changing
# matching, labels or overlaps for nonempty-GT frames.
assert head.train_cfg.assigner.type == "HungarianAssigner3D"


from carla_class_consistent_assigner_11925d13 import (
    class_consistent_assign,
)

original_assign = head.bbox_assigner.assign

audit = {
    "calls": 0,
    "empty_calls": 0,
    "positive": 0,
    "mismatch": 0,
}

def real_class_consistent_assign(*args, **kwargs):
    assert not kwargs
    assert len(args) == 5

    proposals, gt_boxes, gt_labels, cls_pred, train_cfg = args

    assert head.query_labels.shape[0] == 1

    query_labels = head.query_labels[0]

    assert query_labels.numel() == proposals.shape[0]
    assert cls_pred.shape[-1] == proposals.shape[0]

    result = class_consistent_assign(
        original_assign=original_assign,
        query_labels=query_labels,
        bboxes=proposals,
        gt_bboxes=gt_boxes,
        gt_labels=gt_labels,
        cls_pred=cls_pred,
        train_cfg=train_cfg,
    )

    positive = result.gt_inds > 0

    mismatches = int(
        (result.labels[positive] != query_labels[positive])
        .sum().item()
    )

    assert mismatches == 0

    audit["calls"] += 1
    audit["positive"] += int(positive.sum().item())
    audit["mismatch"] += mismatches

    if gt_boxes.shape[0] == 0:
        audit["empty_calls"] += 1

        assert not positive.any().item()
        assert torch.all(result.gt_inds == 0).item()
        assert torch.all(result.labels == -1).item()
        assert torch.all(result.max_overlaps == 0).item()

        print(
            "CLASS_CONSISTENT_EMPTY_GT: PASS",
            flush=True,
        )
    else:
        print(
            "CLASS_CONSISTENT_ASSIGN:",
            "GT=", gt_boxes.shape[0],
            "queries=", proposals.shape[0],
            "positive=", int(positive.sum().item()),
            "mismatch=", mismatches,
            flush=True,
        )

    return result

head.bbox_assigner.assign = real_class_consistent_assign

original_targets = head.get_targets
diagnostics = {}

def consistent_targets(*args, **kwargs):
    result = list(original_targets(*args, **kwargs))

    labels = result[0].reshape(-1)
    query_classes = head.query_labels.reshape(-1)

    weights = result[3]
    flat = weights.clone().reshape(
        -1, weights.shape[-1]
    )

    assert len(labels) == len(query_classes)

    positive = flat[:, 3:6].sum(1) > 0

    mismatch = (
        positive
        & (labels != query_classes)
        & (labels < head.num_classes)
    )

    diagnostics["positive"] = int(positive.sum().item())
    diagnostics["mismatch"] = int(mismatch.sum().item())

    flat[mismatch, 3:6] = 0.0
    result[3] = flat.reshape_as(weights)

    return tuple(result)

head.get_targets = consistent_targets

model.cuda()
training_mode(model)

expected_frozen = {
    "encoders.camera.vtransform.dx",
    "encoders.camera.vtransform.bx",
    "encoders.camera.vtransform.nx",
    "encoders.camera.vtransform.frustum",
}

actual_frozen = {
    name
    for name, param in model.named_parameters()
    if not param.requires_grad
}

assert actual_frozen == expected_frozen, (
    "Unexpected frozen parameters:",
    actual_frozen.symmetric_difference(expected_frozen)
)

total_parameters = sum(
    p.numel() for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    "Total parameters:", total_parameters,
    flush=True
)

print(
    "Trainable parameters:", trainable_parameters,
    flush=True
)

print(
    "Frozen geometric tensors:",
    len(actual_frozen),
    flush=True
)

assert trainable_parameters == 36358814

offset_params = list(head.class_dim_offset.parameters())
offset_ids = {id(p) for p in offset_params}

shared_params = [
    p for p in model.parameters()
    if id(p) not in offset_ids
]

optimizer = torch.optim.AdamW([
    {
        "params": shared_params,
        "lr": LR,
        "weight_decay": 0.01,
    },
    {
        "params": offset_params,
        "lr": LR,
        "weight_decay": 0.0,
    },
])

initial_offset = (
    head.class_dim_offset.weight.detach().clone()
)

print("Optimizer: AdamW", flush=True)
print("Shared LR:", LR, flush=True)
print("Offset LR:", LR, flush=True)
print("Pedestrian NMS exemption: DISABLED", flush=True)
print("Training scope: FULL MODEL", flush=True)

for step, index in enumerate(indices, 1):
    sample = dataset[index]
    assert sample is not None

    batch = prepare(sample)

    training_mode(model)
    optimizer.zero_grad(set_to_none=True)

    outputs = model(**batch)

    assert diagnostics["mismatch"] == 0, (
        "Cross-class positive target detected",
        diagnostics,
    )


    losses = {}

    for key, value in outputs.items():
        name = key.split("/")[-1]

        if "loss" not in name:
            continue

        if isinstance(value, (tuple, list)):
            value = sum(value)

        assert torch.isfinite(value).all(), name
        losses[name] = value

    required = (
        "loss_heatmap",
        "layer_-1_loss_cls",
        "layer_-1_loss_bbox",
    )

    for name in required:
        assert name in losses, (name, list(losses))

    total = sum(losses.values())
    assert torch.isfinite(total).item()

    total.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 35.0
    )

    assert torch.isfinite(grad_norm).item()

    optimizer.step()

    if step % 50 == 0 or step == STEPS:
        checkpoint_path = OUT / f"step{step}.pth"
        temporary_path = OUT / f"step{step}.pth.tmp"

        assert not checkpoint_path.exists()
        assert not temporary_path.exists()

        torch.save({
            "model": {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            },
            "optimizer": optimizer.state_dict(),
            "step": step,
            "training_type": "fullmodel_240_pilot",
            "source_checkpoint": str(SOURCE),
            "training_indices": indices,
            "selected_scenes": selected_scenes,
            "learning_rate": LR,
            "pedestrian_nms_exempt": False,
            "mismatch_dim_mask_enabled": True,
            "seed": SEED,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }, temporary_path)

        os.replace(temporary_path, checkpoint_path)

        print(
            "CHECKPOINT SAVED:",
            checkpoint_path,
            flush=True
        )

    print(
        f"STEP {step}/{STEPS} "
        f"index={index} "
        f"total={total.item():.6f} "
        f"heatmap={losses['loss_heatmap'].item():.6f} "
        f"cls={losses['layer_-1_loss_cls'].item():.6f} "
        f"bbox={losses['layer_-1_loss_bbox'].item():.6f} "
        f"positive={diagnostics['positive']} "
        f"mismatch={diagnostics['mismatch']} "
        f"grad_norm={float(grad_norm):.6f}",
        flush=True
    )

offset_change = (
    head.class_dim_offset.weight.detach()
    - initial_offset
).abs().max().item()

print(
    "Maximum dimension-offset change:",
    offset_change,
    flush=True
)

assert audit["empty_calls"] >= 1, (
    "Empty-GT assignment was not exercised"
)
print(
    "EMPTY_GT_TEST: PASS",
    "assignments=",
    audit["empty_calls"],
    flush=True,
)

print("Training loop completed", flush=True)

head.bbox_assigner.assign = original_assign
head.get_targets = original_targets
dim_handle.remove()

print("LAST CHECKPOINT:", OUT / f"step{STEPS}.pth", flush=True)
print("=== STEP 11.9.24.3 COMPLETE ===", flush=True)
print("Optimizer updates:", STEPS, flush=True)
print("Original DIMCAL checkpoint: UNCHANGED", flush=True)
print("Original source scripts: UNCHANGED", flush=True)
