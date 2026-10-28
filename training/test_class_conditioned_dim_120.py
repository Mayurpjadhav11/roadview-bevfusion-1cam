#!/usr/bin/env python3
"""One-scene diagnostic with BatchNorm frozen during training."""

import csv
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from mmcv.parallel import DataContainer

from carla_model_loader import build_carla_model
from carla_training_dataset import build_carla_training_dataset

SEED = 42
STEPS = 120
LR = 1e-4
CHECKPOINT_STEPS = (20, 40, 60, 80, 100)

def learning_rate_at(step):
    if step <= 80:
        return LR
    if step <= 160:
        return 5e-5
    return 2e-5

INITIAL = Path(
    "/outputs/checkpoints/"
    "single_scene_fit_20261028T093535Z/initial.pth"
)

RUN = datetime.now(timezone.utc).strftime(
    "class_conditioned_dim_120_%Y%m%dT%H%M%SZ"
)

OUT = Path("/outputs/experiments") / RUN
CKPT = Path("/outputs/checkpoints") / RUN

BN = torch.nn.modules.batchnorm._BatchNorm

def disable_dropout(model):
    disabled = []

    for name, module in model.named_modules():
        if isinstance(
            module,
            torch.nn.modules.dropout._DropoutNd
        ):
            if module.p != 0:
                disabled.append((name, module.p))
                module.p = 0.0

        elif isinstance(module, torch.nn.MultiheadAttention):
            if module.dropout != 0:
                disabled.append((name, module.dropout))
                module.dropout = 0.0

    return disabled


MATRICES = [
    "camera2ego", "lidar2ego", "lidar2camera",
    "lidar2image", "camera_intrinsics",
    "camera2lidar", "img_aug_matrix",
    "lidar_aug_matrix",
]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap(x):
    return x.data if isinstance(x, DataContainer) else x


def gpu(x, dtype=torch.float32):
    x = unwrap(x)

    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    return x.to("cuda", dtype=dtype).contiguous()


def prepare(sample, with_gt=True):
    result = {
        "img": gpu(sample["img"]).unsqueeze(0),
        "points": [gpu(sample["points"])],
        "metas": [unwrap(sample["metas"])],
    }

    for key in MATRICES:
        result[key] = gpu(sample[key]).unsqueeze(0)

    if with_gt:
        result["gt_bboxes_3d"] = [
            unwrap(sample["gt_bboxes_3d"]).to("cuda")
        ]
        result["gt_labels_3d"] = [
            gpu(sample["gt_labels_3d"], torch.long)
        ]

    return result


def get_losses(outputs):
    result = {}

    for name, value in outputs.items():
        if "loss" not in name:
            continue

        values = (
            value if isinstance(value, (list, tuple))
            else [value]
        )

        result[name.split("/")[-1]] = sum(
            item.mean() for item in values
        )

    required = [
        "loss_heatmap",
        "layer_-1_loss_cls",
        "layer_-1_loss_bbox",
    ]

    assert all(k in result for k in required)

    return result


def freeze_bn_mode(model):
    # Train the network, but use running stats in every BN.
    model.train()

    for module in model.modules():
        if isinstance(module, BN):
            module.eval()

    active = sum(
        int(module.training)
        for module in model.modules()
        if isinstance(module, BN)
    )

    assert active == 0


print("=" * 65)
print("STEP 11.8.7 — FROZEN BATCHNORM OVERFITTING TEST")
print("=" * 65)

assert INITIAL.is_file()

seed_all(SEED)
torch.backends.cudnn.benchmark = False

dataset = build_carla_training_dataset()

saved = torch.load(INITIAL, map_location="cpu")

assert saved["seed"] == SEED
assert saved["heatmap_prior"] == 0.01

index = int(saved["sample_index"])
sample = dataset[index]

print("Scene:", index)
print("GT objects:", len(unwrap(sample["gt_labels_3d"])))
print("Learning rate:", LR)
print("Updates:", STEPS)

seed_all(SEED)

model, _ = build_carla_model(
    pretrained=False,
    device="cpu",
)

model.load_state_dict(saved["model"], strict=True)
model.cuda()

from carla_dataset_config import CLASSES

CAR = list(CLASSES).index("car")
PEDESTRIAN = list(CLASSES).index("pedestrian")

head = model.heads["object"]
assert head.num_decoder_layers == 1
assert head.num_classes == len(CLASSES)

head.class_dim_offset = torch.nn.Embedding(
    head.num_classes, 3
).cuda()

torch.nn.init.zeros_(head.class_dim_offset.weight)

def dimension_hook(module, inputs, result):
    correction = head.class_dim_offset(
        head.query_labels
    ).transpose(1, 2)

    assert correction.shape == result["dim"].shape

    updated = dict(result)
    updated["dim"] = result["dim"] + correction
    return updated

hook = (
    head.prediction_heads[0]
    .register_forward_hook(dimension_hook)
)

original_targets = head.get_targets
mismatch_count = {"value": 0}

def consistent_targets(*args, **kwargs):
    result = list(
        original_targets(*args, **kwargs)
    )

    labels = result[0].reshape(-1)
    query_labels = head.query_labels.reshape(-1)

    weights = result[3]
    flat_weights = weights.clone().reshape(
        -1, weights.shape[-1]
    )

    assert len(labels) == len(query_labels)
    assert flat_weights.shape[1] >= 6

    positive = flat_weights[:, 3:6].sum(1) > 0

    mismatch = (
        positive
        & (labels != query_labels)
        & (labels < head.num_classes)
    )

    mismatch_count["value"] = int(
        mismatch.sum().item()
    )

    # Only mask dimension regression for
    # initial-class/GT-class mismatches.
    flat_weights[mismatch, 3:6] = 0.0

    result[3] = flat_weights.reshape_as(weights)
    return tuple(result)

head.get_targets = consistent_targets

print("Class-conditioned dimensions enabled", flush=True)
print("Initial dimension offsets: zero", flush=True)

# Ensure the experiment really begins from our
# original 1% heatmap-prior initialization.
bias = dict(
    model.heads["object"].heatmap_head.named_parameters()
)["1.bias"]

assert torch.allclose(
    bias.detach().sigmoid(),
    torch.full_like(bias, 0.01),
    atol=0.001,
)

bn_buffers = {
    name: value.detach().clone()
    for name, value in model.named_buffers()
    if (
        "running_mean" in name
        or "running_var" in name
        or "num_batches_tracked" in name
    )
}

print("Frozen BN buffers:", len(bn_buffers))


def evaluate(stage):
    model.eval()

    with torch.no_grad():
        prediction = model(
            **prepare(sample, with_gt=False)
        )[0]

    pred_xy = (
        prediction["boxes_3d"]
        .tensor[:, :2]
        .detach()
        .cpu()
        .float()
    )

    pred_labels = (
        prediction["labels_3d"]
        .detach()
        .cpu()
        .long()
    )

    scores = (
        prediction["scores_3d"]
        .detach()
        .cpu()
        .float()
    )

    gt_boxes = unwrap(sample["gt_bboxes_3d"])

    gt_xy = gt_boxes.tensor[:, :2].detach().cpu().float()
    gt_labels = unwrap(sample["gt_labels_3d"]).cpu().long()

    distances = torch.cdist(gt_xy, pred_xy)

    closest = distances.min(dim=1).values

    correct_within_2m = 0

    for j, label in enumerate(gt_labels):
        matching = pred_labels == label

        if bool(matching.any()):
            if float(distances[j, matching].min()) <= 2.0:
                correct_within_2m += 1

    model.training = True

    with torch.no_grad():
        outputs = model(
            **prepare(sample, with_gt=True)
        )

    model.training = False

    losses = get_losses(outputs)

    metrics = {
        "stage": stage,
        "heatmap": float(losses["loss_heatmap"]),
        "classification": float(
            losses["layer_-1_loss_cls"]
        ),
        "bbox": float(losses["layer_-1_loss_bbox"]),
        "mean_nearest_xy": float(closest.mean()),
        "any_within_2m": int((closest <= 2).sum()),
        "same_class_within_2m": correct_within_2m,
        "max_score": float(scores.max()),
    }

    print(f"\n=== {stage.upper()} ===")

    for name, value in metrics.items():
        if name != "stage":
            print(f"{name}: {value}")

    return metrics


before = evaluate("before")

print("\n=== TRAIN WITH FROZEN BATCHNORM ===")

offset_params = list(
    head.class_dim_offset.parameters()
)
offset_ids = {id(p) for p in offset_params}

backbone_params = [
    p for p in model.parameters()
    if id(p) not in offset_ids
]

optimizer = torch.optim.AdamW(
    [
        {
            "params": backbone_params,
            "lr": LR,
            "weight_decay": 0.01,
            "is_offset": False,
        },
        {
            "params": offset_params,
            "lr": LR * 200,
            "weight_decay": 0.0,
            "is_offset": True,
        },
    ],
)

CKPT.mkdir(parents=True, exist_ok=False)
records = []

seed_all(SEED + 100)

for step in range(1, STEPS + 1):
    freeze_bn_mode(model)
    disabled = disable_dropout(model)
    if step == 1:
        print('Dropout operations disabled:', disabled, flush=True)
        assert len(disabled) > 0, 'No dropout operations found'

    current_lr = learning_rate_at(step)

    for group in optimizer.param_groups:
        group["lr"] = (
            current_lr * 200
            if group["is_offset"]
            else current_lr
        )

    optimizer.zero_grad(set_to_none=True)

    outputs = model(**prepare(sample))
    losses = get_losses(outputs)

    total = sum(losses.values())

    assert torch.isfinite(total)

    total.backward()

    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 35.0
    )

    assert torch.isfinite(norm)

    optimizer.step()

    row = {
        "step": step,
        "mismatched_positives": mismatch_count["value"],
        "learning_rate": current_lr,
        "grad_norm_preclip": float(norm),
        "heatmap": float(
            losses["loss_heatmap"].detach()
        ),
        "classification": float(
            losses["layer_-1_loss_cls"].detach()
        ),
        "bbox": float(
            losses["layer_-1_loss_bbox"].detach()
        ),
        "total": float(total.detach()),
    }

    records.append(row)

    if step == 1 or step % 20 == 0:
        offsets = (
            head.class_dim_offset.weight
            .detach().cpu()
        )
        print(
            f"CLASS OFFSETS step={step} "
            f"car={offsets[CAR].tolist()} "
            f"pedestrian={offsets[PEDESTRIAN].tolist()} "
            f"mismatched={mismatch_count['value']}",
            flush=True
        )

    if step in CHECKPOINT_STEPS:
        path = CKPT / f"step{step}.pth"

        assert not path.exists()

        torch.save(
            {
                "model": {
                    k: v.detach().cpu()
                    for k, v in model.state_dict().items()
                },
                "step": step,
                "sample_index": index,
                "frozen_bn": True,
                "dropout_disabled": True,
                "learning_rate": current_lr,
                "initial_checkpoint": str(INITIAL),
            },
            path,
        )

        print(
            f"CHECKPOINT SAVED: {path}",
            flush=True,
        )

    if step == 1 or step % 10 == 0:
        print(
            f"Step {step:03d}/{STEPS} | "
            f"heatmap={row['heatmap']:.4f} | "
            f"cls={row['classification']:.4f} | "
            f"bbox={row['bbox']:.4f} | "
            f"total={row['total']:.4f}",
            flush=True,
        )

after = evaluate("after")

print("\n=== VERIFY BN RUNNING STATISTICS ===")

changed = [
    name for name, buffer in model.named_buffers()
    if name in bn_buffers
    and not torch.equal(buffer, bn_buffers[name])
]

print("Changed BN buffers:", changed)

assert not changed, (
    "BatchNorm buffers changed during frozen-BN training"
)

OUT.mkdir(parents=True, exist_ok=False)

with (OUT / "training.csv").open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(records[0])
    )
    writer.writeheader()
    writer.writerows(records)

with (OUT / "evaluation.csv").open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(before)
    )
    writer.writeheader()
    writer.writerows([before, after])

torch.save(
    {
        "model": {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
        },
        "step": STEPS,
        "sample_index": index,
        "frozen_bn": True,
        "dropout_disabled": True,
        "learning_rate": learning_rate_at(STEPS),
        "class_dim_offset_enabled": True,
        "offset_learning_rate_multiplier": 200,
        "mismatch_dim_mask_enabled": True,
        "initial_checkpoint": str(INITIAL),
    },
    CKPT / "step120.pth",
)

print("\nExperiment directory:", OUT)
print("Checkpoint directory:", CKPT)

print("\n" + "=" * 65)
print("FROZEN BATCHNORM TEST COMPLETE")
print("=" * 65)
print("Existing checkpoints preserved.")
print("NVIDIA source unchanged.")
