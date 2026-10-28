#!/usr/bin/env python3
"""Fixed 50-scene before/after validation for heatmap A/B."""

import csv
import random
from pathlib import Path

import numpy as np
import torch
from mmcv.parallel import DataContainer

from carla_model_loader import build_carla_model
from carla_dataset_config import build_dataset
from carla_heatmap_initialization import apply_heatmap_prior

RUN = "heatmap_ab_20261028T091710Z"

CKPT = Path("/outputs/checkpoints") / RUN
OUT = Path("/outputs/experiments") / RUN
REPORT = OUT / "validation_50_before_after.csv"

SEED = 42
NUM_SCENES = 50

METRICS = [
    "loss_heatmap",
    "layer_-1_loss_cls",
    "layer_-1_loss_bbox",
]


def unwrap(value):
    return (
        value.data
        if isinstance(value, DataContainer)
        else value
    )


def gpu(value, dtype=torch.float32):
    value = unwrap(value)

    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)

    return value.to(
        device="cuda",
        dtype=dtype,
    ).contiguous()


def prepare(sample):
    matrices = [
        "camera2ego",
        "lidar2ego",
        "lidar2camera",
        "lidar2image",
        "camera_intrinsics",
        "camera2lidar",
        "img_aug_matrix",
        "lidar_aug_matrix",
    ]

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

    for name in matrices:
        result[name] = gpu(sample[name]).unsqueeze(0)

    return result


def get_losses(outputs):
    result = {}

    for name, value in outputs.items():
        if "loss" not in name:
            continue

        values = (
            value
            if isinstance(value, (list, tuple))
            else [value]
        )

        result[name.split("/")[-1]] = sum(
            float(item.detach().float().mean().cpu())
            for item in values
        )

    assert all(k in result for k in METRICS), list(result)

    assert all(np.isfinite(result[k]) for k in METRICS)

    return result


print("=" * 65)
print("STEP 11.7.4 — CONTROLLED 50-SCENE A/B VALIDATION")
print("=" * 65)

assert (CKPT / "exact_initial.pth").is_file()
assert (CKPT / "default_step20.pth").is_file()
assert (CKPT / "prior_1percent_step20.pth").is_file()
assert not REPORT.exists(), "Report already exists"

print("\n=== 1. CACHE FIXED VALIDATION SCENES ===")

dataset = build_dataset("val")

assert len(dataset) == 1800

indices = random.Random(SEED).sample(
    range(len(dataset)),
    len(dataset),
)

samples = []

for index in indices:
    sample = dataset[index]

    if len(unwrap(sample["gt_labels_3d"])) == 0:
        continue

    samples.append((index, sample))

    if len(samples) == NUM_SCENES:
        break

assert len(samples) == NUM_SCENES

print("Scenes:", [i for i, _ in samples])
print("Number of scenes:", len(samples))


print("\n=== 2. LOAD EXACT INITIAL CHECKPOINT ===")

initial_checkpoint = torch.load(
    CKPT / "exact_initial.pth",
    map_location="cpu",
)

assert initial_checkpoint["seed"] == SEED

initial_state = initial_checkpoint["model"]

torch.manual_seed(SEED)

model, _ = build_carla_model(
    pretrained=False,
    device="cpu",
)

model.cuda()

rows = []


def evaluate(variant, stage):
    model.eval()

    # Keep child modules in evaluation mode, but
    # request the training-loss branch from the root.
    model.training = True

    for index, sample in samples:
        with torch.no_grad():
            output = model(**prepare(sample))

        values = get_losses(output)

        row = {
            "variant": variant,
            "stage": stage,
            "sample_index": index,
            "num_objects": len(
                unwrap(sample["gt_labels_3d"])
            ),
        }

        for metric in METRICS:
            row[metric] = values[metric]

        row["total"] = sum(
            row[m] for m in METRICS
        )

        rows.append(row)

    subset = [
        r for r in rows
        if r["variant"] == variant
        and r["stage"] == stage
    ]

    print(
        f"{variant:<16} {stage:<6} | "
        f"heatmap={np.mean([r['loss_heatmap'] for r in subset]):.4f} | "
        f"cls={np.mean([r['layer_-1_loss_cls'] for r in subset]):.4f} | "
        f"bbox={np.mean([r['layer_-1_loss_bbox'] for r in subset]):.4f} | "
        f"total={np.mean([r['total'] for r in subset]):.4f}",
        flush=True,
    )


print("\n=== 3. EVALUATE EXACT STARTING STATES ===")

for variant in ["default", "prior_1percent"]:
    model.load_state_dict(initial_state, strict=True)

    if variant == "prior_1percent":
        apply_heatmap_prior(model, 0.01)

    evaluate(variant, "before")

    trained = torch.load(
        CKPT / f"{variant}_step20.pth",
        map_location="cpu",
    )

    assert trained["variant"] == variant
    assert trained["step"] == 20
    assert trained["seed"] == SEED

    model.load_state_dict(
        trained["model"],
        strict=True,
    )

    del trained

    evaluate(variant, "after")


print("\n=== 4. COUNT IMPROVED VALIDATION SCENES ===")

for variant in ["default", "prior_1percent"]:
    before = {
        r["sample_index"]: r
        for r in rows
        if r["variant"] == variant
        and r["stage"] == "before"
    }

    after = {
        r["sample_index"]: r
        for r in rows
        if r["variant"] == variant
        and r["stage"] == "after"
    }

    assert len(before) == len(after) == NUM_SCENES
    assert set(before) == set(after)

    print(f"\nVariant: {variant}")

    for metric in METRICS + ["total"]:
        changes = [
            after[i][metric] - before[i][metric]
            for i in before
        ]

        improved = sum(v < -1e-6 for v in changes)
        worsened = sum(v > 1e-6 for v in changes)

        print(
            f"{metric:<24} "
            f"improved={improved:02d} "
            f"worsened={worsened:02d} "
            f"mean_delta={np.mean(changes):+.4f}"
        )


print("\n=== 5. SAVE REPORT ===")

with REPORT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys()),
    )

    writer.writeheader()
    writer.writerows(rows)

print("Report:", REPORT)

print("\n" + "=" * 65)
print("50-SCENE A/B VALIDATION COMPLETE")
print("=" * 65)
print("Four model states evaluated.")
print("No training performed.")
print("Existing checkpoints preserved.")
