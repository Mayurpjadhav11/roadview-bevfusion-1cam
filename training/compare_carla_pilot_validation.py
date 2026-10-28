#!/usr/bin/env python3
"""Compare initial and 20-step BEVFusion models on fixed CARLA scenes."""

import csv
import random
from pathlib import Path

import numpy as np
import torch
from mmcv.parallel import DataContainer

from carla_dataset_config import build_dataset
from carla_model_loader import build_carla_model


SEED = 42
NUM_SAMPLES = 6

RUN = "carla_pilot_20steps_20261028T081055Z"

CHECKPOINT = (
    Path("/outputs/checkpoints") / RUN / "pilot_last.pth"
)

REPORT = (
    Path("/outputs/experiments") / RUN /
    "fixed_validation_comparison.csv"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap(value):
    return (
        value.data
        if isinstance(value, DataContainer)
        else value
    )


def gpu_tensor(value, dtype=torch.float32):
    value = unwrap(value)

    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)

    return value.to(
        device="cuda",
        dtype=dtype,
    ).contiguous()


def prepare_sample(sample):
    image = gpu_tensor(sample["img"]).unsqueeze(0)
    points = [gpu_tensor(sample["points"])]

    matrix_keys = [
        "camera2ego",
        "lidar2ego",
        "lidar2camera",
        "lidar2image",
        "camera_intrinsics",
        "camera2lidar",
        "img_aug_matrix",
        "lidar_aug_matrix",
    ]

    matrices = {
        key: gpu_tensor(sample[key]).unsqueeze(0)
        for key in matrix_keys
    }

    boxes = unwrap(sample["gt_bboxes_3d"]).to("cuda")

    labels = gpu_tensor(
        sample["gt_labels_3d"],
        dtype=torch.long,
    )

    metadata = unwrap(sample["metas"])

    assert isinstance(metadata, dict)
    assert len(boxes) == len(labels)
    assert len(labels) > 0

    return {
        "img": image,
        "points": points,
        **matrices,
        "metas": [metadata],
        "gt_bboxes_3d": [boxes],
        "gt_labels_3d": [labels],
    }


print("=" * 65)
print("STEP 11.6.2 — FIXED CARLA VALIDATION COMPARISON")
print("=" * 65)

assert CHECKPOINT.is_file()

print("\n=== 1. SELECT FIXED VALIDATION SCENES ===")

dataset = build_dataset("val")

assert len(dataset) == 1800

# Cache the actual processed samples to ensure both
# models receive identical camera/LiDAR inputs.
samples = []

for index in range(len(dataset)):
    sample = dataset[index]

    labels = unwrap(sample["gt_labels_3d"])

    if len(labels) == 0:
        continue

    samples.append((index, sample))

    if len(samples) == NUM_SAMPLES:
        break

assert len(samples) == NUM_SAMPLES

print("Selected sample indices:", [i for i, _ in samples])
print("Samples cached:", len(samples))


def evaluate_model(name, checkpoint=None):
    print(f"\n=== 2. EVALUATE {name.upper()} ===")

    # Reconstruct the seed-42 initialization used for
    # the original pilot. The initial model was not saved,
    # so this is a seeded reconstruction, not a restored
    # initial checkpoint.
    set_seed(SEED)

    model, _ = build_carla_model(
        pretrained=True,
        device="cuda",
    )

    if checkpoint is not None:
        saved = torch.load(
            str(checkpoint),
            map_location="cpu",
        )

        model.load_state_dict(
            saved["model"],
            strict=True,
        )

        print("Restored checkpoint step:", saved["step"])

    # Keep BatchNorm and dropout in evaluation mode.
    model.eval()

    # BEVFusion.forward_single uses the root model's
    # training flag to choose its loss-returning branch.
    # Set only that flag to True, leaving all children
    # in evaluation mode for reproducible validation.
    model.training = True

    assert model.training is True

    results = []

    for index, sample in samples:
        inputs = prepare_sample(sample)

        with torch.no_grad():
            outputs = model(**inputs)

        assert isinstance(outputs, dict)

        record = {
            "model": name,
            "sample_index": index,
            "num_objects": len(inputs["gt_labels_3d"][0]),
        }

        for key, value in outputs.items():
            if "loss" not in key:
                continue

            values = (
                value if isinstance(value, (list, tuple))
                else [value]
            )

            numeric = sum(
                float(v.detach().float().mean().cpu())
                for v in values
            )

            assert np.isfinite(numeric), key

            # Remove train/stats prefix for consistent
            # output column naming.
            short_key = key.split("/", 2)[-1]

            record[short_key] = numeric

        expected = [
            "loss_heatmap",
            "layer_-1_loss_cls",
            "layer_-1_loss_bbox",
        ]

        for term in expected:
            assert term in record, (
                f"Missing {term}: {list(outputs)}"
            )

        record["total_loss"] = sum(
            record[term] for term in expected
        )

        results.append(record)

        print(
            f"{name:>8} | "
            f"sample={index:04d} | "
            f"objects={record['num_objects']:2d} | "
            f"heatmap={record['loss_heatmap']:.2f} | "
            f"cls={record['layer_-1_loss_cls']:.2f} | "
            f"bbox={record['layer_-1_loss_bbox']:.2f} | "
            f"total={record['total_loss']:.2f}",
            flush=True,
        )

    del model
    torch.cuda.empty_cache()

    return results


print("\n=== 3. COMPARE MODEL STATES ===")

initial_results = evaluate_model("initial")

trained_results = evaluate_model(
    "step20",
    checkpoint=CHECKPOINT,
)

print("\n=== 4. SUMMARY ===")

metrics = [
    "loss_heatmap",
    "layer_-1_loss_cls",
    "layer_-1_loss_bbox",
    "total_loss",
]

for metric in metrics:
    before = np.mean([
        row[metric] for row in initial_results
    ])

    after = np.mean([
        row[metric] for row in trained_results
    ])

    change = 100 * (after - before) / max(abs(before), 1e-12)

    print(
        f"{metric:<24} "
        f"initial={before:12.4f} "
        f"step20={after:12.4f} "
        f"change={change:+9.2f}%"
    )

print("\n=== 5. SAVE COMPARISON ===")

assert not REPORT.exists(), (
    "Report already exists; refusing to overwrite"
)

all_rows = initial_results + trained_results

with REPORT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(all_rows[0].keys()),
    )

    writer.writeheader()
    writer.writerows(all_rows)

print("Report:", REPORT)

print("\n" + "=" * 65)
print("FIXED CARLA VALIDATION COMPARISON PASSED")
print("=" * 65)
print("Same six processed scenes for both models.")
print("No optimizer updates performed.")
print("No checkpoint modified.")
