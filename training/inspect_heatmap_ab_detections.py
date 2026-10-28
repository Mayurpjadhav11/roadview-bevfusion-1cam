#!/usr/bin/env python3
"""Inspect actual 3D predictions from the heatmap A/B checkpoints."""

import csv
from pathlib import Path

import numpy as np
import torch
from mmcv.parallel import DataContainer

from carla_model_loader import build_carla_model
from carla_dataset_config import build_dataset, CLASSES
from carla_heatmap_initialization import apply_heatmap_prior

RUN = "heatmap_ab_20261028T091710Z"

CKPT = Path("/outputs/checkpoints") / RUN
OUT = Path("/outputs/experiments") / RUN

SOURCE = OUT / "validation_50_before_after.csv"
REPORT = OUT / "detection_sanity_10.csv"

NUM_SCENES = 10

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


def unwrap(value):
    return (
        value.data
        if isinstance(value, DataContainer)
        else value
    )


def gpu(value):
    value = unwrap(value)
    return value.to("cuda").contiguous()


def prepare(sample):
    result = {
        "img": gpu(sample["img"]).unsqueeze(0),
        "points": [gpu(sample["points"])],
        "metas": [unwrap(sample["metas"])],
    }

    for key in MATRICES:
        result[key] = gpu(sample[key]).unsqueeze(0)

    return result


print("=" * 65)
print("STEP 11.7.5 — ACTUAL 3D DETECTION SANITY CHECK")
print("=" * 65)

assert SOURCE.is_file()
assert not REPORT.exists(), "Detection report already exists"

with SOURCE.open(newline="") as f:
    comparison = list(csv.DictReader(f))

indices = []

for row in comparison:
    index = int(row["sample_index"])

    if index not in indices:
        indices.append(index)

    if len(indices) == NUM_SCENES:
        break

assert len(indices) == NUM_SCENES

dataset = build_dataset("val")

samples = [(index, dataset[index]) for index in indices]

print("\nValidation indices:", indices)

initial = torch.load(
    CKPT / "exact_initial.pth",
    map_location="cpu",
)

torch.manual_seed(42)

model, _ = build_carla_model(
    pretrained=False,
    device="cpu",
)

model.cuda()

records = []

states = [
    ("default", "before", None),
    ("default", "after", CKPT / "default_step20.pth"),
    ("prior_1percent", "before", None),
    (
        "prior_1percent",
        "after",
        CKPT / "prior_1percent_step20.pth",
    ),
]

for variant, stage, checkpoint_path in states:

    print(f"\n=== {variant} — {stage} ===", flush=True)

    model.load_state_dict(
        initial["model"],
        strict=True,
    )

    if variant == "prior_1percent":
        apply_heatmap_prior(model, 0.01)

    if checkpoint_path is not None:
        saved = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        assert saved["variant"] == variant
        assert saved["step"] == 20

        model.load_state_dict(
            saved["model"],
            strict=True,
        )

        del saved

    model.eval()

    for index, sample in samples:

        with torch.no_grad():
            predictions = model(**prepare(sample))

        assert isinstance(predictions, (list, tuple)), (
            f"Unexpected prediction container: {type(predictions)}"
        )

        assert len(predictions) == 1

        prediction = predictions[0]

        assert isinstance(prediction, dict), (
            f"Unexpected prediction type: {type(prediction)}"
        )

        required = {
            "boxes_3d",
            "scores_3d",
            "labels_3d",
        }

        assert required.issubset(prediction), (
            f"Prediction keys: {list(prediction)}"
        )

        scores = (
            prediction["scores_3d"]
            .detach()
            .float()
            .cpu()
        )

        labels = (
            prediction["labels_3d"]
            .detach()
            .long()
            .cpu()
        )

        boxes = prediction["boxes_3d"]

        assert len(scores) == len(labels) == len(boxes)
        assert bool(torch.isfinite(scores).all())

        gt_labels = unwrap(sample["gt_labels_3d"])
        gt_count = len(gt_labels)

        record = {
            "variant": variant,
            "stage": stage,
            "scene": index,
            "ground_truth_objects": gt_count,
            "predicted_boxes": len(scores),
            "score_mean": (
                scores.mean().item() if len(scores) else 0.0
            ),
            "score_max": (
                scores.max().item() if len(scores) else 0.0
            ),
        }

        for threshold in [0.01, 0.1, 0.3, 0.5]:
            key = f"boxes_ge_{threshold}"
            record[key] = int(
                (scores >= threshold).sum().item()
            )

        records.append(record)

        top = torch.argsort(
            scores, descending=True
        )[:5]

        top_detections = []

        for j in top.tolist():
            class_id = int(labels[j])
            class_name = (
                CLASSES[class_id]
                if 0 <= class_id < len(CLASSES)
                else f"unknown_{class_id}"
            )

            top_detections.append(
                f"{class_name}:{scores[j].item():.3f}"
            )

        print(
            f"Scene {index:04d} | "
            f"GT={gt_count:2d} | "
            f"pred={len(scores):3d} | "
            f">=0.1={record['boxes_ge_0.1']:3d} | "
            f">=0.3={record['boxes_ge_0.3']:3d} | "
            f">=0.5={record['boxes_ge_0.5']:3d} | "
            f"top5={','.join(top_detections)}",
            flush=True,
        )

    subset = [
        r for r in records
        if r["variant"] == variant
        and r["stage"] == stage
    ]

    print(
        "SUMMARY | "
        f"mean_max_score="
        f"{np.mean([r['score_max'] for r in subset]):.5f} | "
        f"mean_boxes_ge_0.3="
        f"{np.mean([r['boxes_ge_0.3'] for r in subset]):.2f}",
        flush=True,
    )


with REPORT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(records[0]),
    )
    writer.writeheader()
    writer.writerows(records)

print("\n" + "=" * 65)
print("3D DETECTION SANITY CHECK COMPLETE")
print("=" * 65)
print("Report:", REPORT)
print("No training performed.")
print("No checkpoints modified.")
