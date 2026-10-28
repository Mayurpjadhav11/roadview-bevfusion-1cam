#!/usr/bin/env python3
"""Controlled one-scene BEVFusion overfitting diagnostic."""

import csv
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from mmcv.parallel import DataContainer

from carla_model_loader import build_carla_model
from carla_training_dataset import build_carla_training_dataset
from carla_heatmap_initialization import apply_heatmap_prior
from carla_dataset_config import CLASSES


SEED = 42
STEPS = 120
LR = 1e-4

RUN = datetime.now(timezone.utc).strftime(
    "single_scene_fit_%Y%m%dT%H%M%SZ"
)

OUT = Path("/outputs/experiments") / RUN
CKPT = Path("/outputs/checkpoints") / RUN

MATRICES = [
    "camera2ego",
    "lidar2ego",
    "lidar2camera",
    "lidar2image",
    "camera_intrinsics",
    "camera2lidar",
    "img_aug_matrix",
    "lidar_aug_matrix",
]

LOSS_NAMES = [
    "loss_heatmap",
    "layer_-1_loss_cls",
    "layer_-1_loss_bbox",
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


def prepare(sample, include_gt=True):
    result = {
        "img": gpu(sample["img"]).unsqueeze(0),
        "points": [gpu(sample["points"])],
        "metas": [unwrap(sample["metas"])],
    }

    for key in MATRICES:
        result[key] = gpu(sample[key]).unsqueeze(0)

    if include_gt:
        result["gt_bboxes_3d"] = [
            unwrap(sample["gt_bboxes_3d"]).to("cuda")
        ]
        result["gt_labels_3d"] = [
            gpu(sample["gt_labels_3d"], torch.long)
        ]

    return result


def extract_losses(outputs):
    result = {}

    for name, value in outputs.items():
        if "loss" not in name:
            continue

        values = (
            value if isinstance(value, (list, tuple))
            else [value]
        )

        result[name.split("/")[-1]] = sum(
            v.mean() for v in values
        )

    assert all(k in result for k in LOSS_NAMES), list(result)
    return result


print("=" * 65)
print("STEP 11.8.1 — ONE-SCENE CARLA OVERFITTING TEST")
print("=" * 65)

seed_all(SEED)
torch.backends.cudnn.benchmark = False

print("\n=== 1. SELECT ONE TRAINING SCENE ===")

dataset = build_carla_training_dataset()
assert dataset.filter_empty_gt is True

selected = None

for index in range(min(100, len(dataset))):
    sample = dataset[index]

    labels = unwrap(sample["gt_labels_3d"])

    if len(labels) >= 4 and len(torch.unique(labels)) >= 2:
        selected = (index, sample)
        break

assert selected is not None, (
    "No suitable multi-class scene found in first 100 samples"
)

index, sample = selected
gt_labels = unwrap(sample["gt_labels_3d"]).long()

print("Selected train index:", index)
print("GT objects:", len(gt_labels))
print(
    "Classes:",
    [CLASSES[int(x)] for x in gt_labels.tolist()]
)

print("\n=== 2. BUILD INITIAL MODEL ===")

seed_all(SEED)

model, _ = build_carla_model(
    pretrained=True,
    device="cuda",
)

bias = apply_heatmap_prior(model, 0.01)
print("Heatmap prior logit:", bias)

OUT.mkdir(parents=True, exist_ok=False)
CKPT.mkdir(parents=True, exist_ok=False)

torch.save(
    {
        "model": {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        },
        "seed": SEED,
        "sample_index": index,
        "heatmap_prior": 0.01,
    },
    CKPT / "initial.pth",
)


def evaluate(stage):
    model.eval()

    with torch.no_grad():
        predictions = model(
            **prepare(sample, include_gt=False)
        )

    assert len(predictions) == 1
    pred = predictions[0]

    pred_xy = pred["boxes_3d"].tensor[:, :2].detach().cpu()
    pred_cls = pred["labels_3d"].detach().long().cpu()
    pred_scores = pred["scores_3d"].detach().float().cpu()

    gt_xy = (
        unwrap(sample["gt_bboxes_3d"])
        .tensor[:, :2]
        .detach()
        .cpu()
    )

    gt_cls = gt_labels.cpu()

    assert len(pred_xy) > 0
    assert torch.isfinite(pred_xy).all()

    dist = torch.cdist(gt_xy.float(), pred_xy.float())

    closest = dist.min(dim=1).values

    same_class = []

    for i in range(len(gt_cls)):
        mask = pred_cls == gt_cls[i]

        if mask.any():
            same_class.append(dist[i, mask].min().item())
        else:
            same_class.append(float("inf"))

    # Request loss only from root model while
    # keeping child modules in eval mode.
    model.training = True

    with torch.no_grad():
        outputs = model(
            **prepare(sample, include_gt=True)
        )

    model.training = False

    parts = extract_losses(outputs)

    result = {
        "stage": stage,
        "heatmap": float(parts["loss_heatmap"].cpu()),
        "classification": float(
            parts["layer_-1_loss_cls"].cpu()
        ),
        "bbox": float(parts["layer_-1_loss_bbox"].cpu()),
        "nearest_xy_mean": closest.mean().item(),
        "any_within_2m": int((closest <= 2).sum()),
        "any_within_4m": int((closest <= 4).sum()),
        "same_class_within_2m": sum(
            d <= 2 for d in same_class
        ),
        "same_class_within_4m": sum(
            d <= 4 for d in same_class
        ),
        "max_score": pred_scores.max().item(),
        "boxes_over_01": int((pred_scores >= 0.1).sum()),
    }

    result["total"] = sum(
        result[k] for k in [
            "heatmap", "classification", "bbox"
        ]
    )

    print(f"\n=== EVALUATION: {stage} ===")

    for key, value in result.items():
        if key != "stage":
            print(f"{key}: {value}")

    print("GT object count:", len(gt_cls))
    print("Predicted proposal count:", len(pred_xy))

    return result


before = evaluate("before")

print("\n=== 3. TRAIN REPEATEDLY ON SAME SCENE ===")

model.train()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0.01,
)

records = []

for step in range(1, STEPS + 1):
    optimizer.zero_grad(set_to_none=True)

    outputs = model(**prepare(sample, include_gt=True))
    parts = extract_losses(outputs)

    total = sum(parts[k] for k in LOSS_NAMES)

    assert torch.isfinite(total), (
        f"Non-finite loss at step {step}"
    )

    total.backward()

    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=35.0,
    )

    assert torch.isfinite(norm), (
        f"Non-finite gradient at step {step}"
    )

    optimizer.step()

    row = {
        "step": step,
        "heatmap": float(
            parts["loss_heatmap"].detach().cpu()
        ),
        "classification": float(
            parts["layer_-1_loss_cls"].detach().cpu()
        ),
        "bbox": float(
            parts["layer_-1_loss_bbox"].detach().cpu()
        ),
        "total": float(total.detach().cpu()),
    }

    records.append(row)

    if step == 1 or step % 20 == 0:
        print(
            f"Step {step:03d}/{STEPS} | "
            f"heatmap={row['heatmap']:.4f} | "
            f"cls={row['classification']:.4f} | "
            f"bbox={row['bbox']:.4f} | "
            f"total={row['total']:.4f}",
            flush=True,
        )

print("\n=== 4. RE-EVALUATE SAME SCENE ===")

after = evaluate("after")

print("\n=== 5. SAVE RESULTS ===")

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
        "optimizer": optimizer.state_dict(),
        "step": STEPS,
        "sample_index": index,
        "seed": SEED,
        "heatmap_prior": 0.01,
    },
    CKPT / "step120.pth",
)

print("Experiment directory:", OUT)
print("Checkpoint directory:", CKPT)

print("\n" + "=" * 65)
print("ONE-SCENE OVERFITTING EXPERIMENT COMPLETE")
print("=" * 65)
print("No previous checkpoints overwritten.")
print("NVIDIA source unchanged.")
