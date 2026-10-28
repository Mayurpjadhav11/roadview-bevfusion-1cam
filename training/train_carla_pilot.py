#!/usr/bin/env python3
"""Reproducible 20-step CARLA BEVFusion training pilot."""

import csv
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from mmcv.parallel import DataContainer

from carla_dataset_config import build_dataset, CLASSES
from carla_model_loader import build_carla_model


SEED = 42
NUM_STEPS = 20
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
GRAD_CLIP = 35.0

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

torch.backends.cudnn.benchmark = False

RUN_ID = datetime.now(timezone.utc).strftime(
    "carla_pilot_20steps_%Y%m%dT%H%M%SZ"
)

EXPERIMENT_DIR = Path("/outputs/experiments") / RUN_ID
CHECKPOINT_DIR = Path("/outputs/checkpoints") / RUN_ID

print("=" * 65)
print("STEP 11.5.3 — CARLA TRAINING PILOT")
print("=" * 65)
print("Run ID:", RUN_ID)
print("Steps:", NUM_STEPS)
print("Seed:", SEED)


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

    assert image.shape == (1, 1, 3, 256, 704)
    assert points[0].shape[1] == 5
    assert len(boxes) == len(labels)
    assert len(labels) > 0
    assert isinstance(metadata, dict)
    assert int(labels.min()) >= 0
    assert int(labels.max()) < 6
    assert torch.isfinite(boxes.tensor).all()

    return {
        "img": image,
        "points": points,
        **matrices,
        "metas": [metadata],
        "gt_bboxes_3d": [boxes],
        "gt_labels_3d": [labels],
    }


print("\n=== 1. LOAD DATASET ===")

dataset = build_dataset("train")

assert len(dataset) == 16200
assert len(CLASSES) == 6

# Use separate scenes instead of updating repeatedly
# on one fixed example. Only the pilot excludes empty scenes.
rng = random.Random(SEED)
candidates = list(range(len(dataset)))
rng.shuffle(candidates)

indices = []

for index in candidates:
    annotations = dataset.get_ann_info(index)

    if len(annotations["gt_labels_3d"]) > 0:
        indices.append(index)

    if len(indices) == NUM_STEPS:
        break

assert len(indices) == NUM_STEPS

print("Training dataset:", len(dataset))
print("Selected nonempty samples:", len(indices))
print("Selected indices:", indices)


print("\n=== 2. BUILD PRETRAINED MODEL ===")

model, cfg = build_carla_model(
    pretrained=True,
    device="cuda",
)

model.train()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

print("Model:", type(model).__name__)
print("Optimizer: AdamW")
print("Learning rate:", LEARNING_RATE)
print("Gradient clipping:", GRAD_CLIP)

torch.cuda.reset_peak_memory_stats()


print("\n=== 3. EXECUTE TRAINING STEPS ===")

metrics = []

for step, index in enumerate(indices, start=1):
    start = time.perf_counter()

    sample = dataset[index]
    inputs = prepare_sample(sample)

    optimizer.zero_grad(set_to_none=True)

    outputs = model(**inputs)

    loss_terms = {}

    for name, value in outputs.items():
        if not name.startswith("loss/"):
            continue

        values = (
            value
            if isinstance(value, (tuple, list))
            else [value]
        )

        loss_terms[name] = sum(
            item.mean() for item in values
        )

    assert loss_terms, "No differentiable losses returned"

    for name, loss in loss_terms.items():
        assert torch.isfinite(loss).all(), name
        assert loss.requires_grad, name

    total_loss = sum(loss_terms.values())

    assert torch.isfinite(total_loss).all()

    total_loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=GRAD_CLIP,
    )

    assert torch.isfinite(grad_norm), (
        "Non-finite gradient norm"
    )

    optimizer.step()
    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    record = {
        "step": step,
        "sample_index": index,
        "num_objects": len(inputs["gt_labels_3d"][0]),
        "loss_total": float(total_loss.detach().cpu()),
        "gradient_norm_before_clipping": float(grad_norm),
        "elapsed_seconds": elapsed,
    }

    for name, value in loss_terms.items():
        record[name] = float(value.detach().cpu())

    metrics.append(record)

    print(
        f"Step {step:02d}/{NUM_STEPS} | "
        f"sample={index} | "
        f"objects={record['num_objects']} | "
        f"loss={record['loss_total']:.4f} | "
        f"grad={record['gradient_norm_before_clipping']:.2f} | "
        f"time={elapsed:.3f}s",
        flush=True,
    )


print("\n=== 4. VERIFY COMPLETED PILOT ===")

assert len(metrics) == NUM_STEPS
assert len(set(indices)) == NUM_STEPS

first_loss = metrics[0]["loss_total"]
last_loss = metrics[-1]["loss_total"]

print("First batch loss:", first_loss)
print("Last batch loss:", last_loss)

print(
    "Peak allocated VRAM: {:.1f} MiB".format(
        torch.cuda.max_memory_allocated() / 1024**2
    )
)
print(
    "Peak reserved VRAM: {:.1f} MiB".format(
        torch.cuda.max_memory_reserved() / 1024**2
    )
)

print(
    "Note: Different batches have different difficulties."
)
print(
    "First-to-last loss is not proof of learning."
)


print("\n=== 5. SAVE METRICS AND CHECKPOINT ===")

EXPERIMENT_DIR.mkdir(parents=True, exist_ok=False)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=False)

metrics_path = EXPERIMENT_DIR / "metrics.csv"
summary_path = EXPERIMENT_DIR / "summary.json"
checkpoint_path = CHECKPOINT_DIR / "pilot_last.pth"

with metrics_path.open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(metrics[0].keys()),
    )

    writer.writeheader()
    writer.writerows(metrics)

summary = {
    "run_id": RUN_ID,
    "seed": SEED,
    "steps": NUM_STEPS,
    "dataset": "CARLA",
    "classes": list(CLASSES),
    "batch_size": 1,
    "optimizer": "AdamW",
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "gradient_clip_max_norm": GRAD_CLIP,
    "pretrained_initialization": "ImageNet ResNet50 only",
    "first_batch_loss": first_loss,
    "last_batch_loss": last_loss,
    "sample_indices": indices,
    "peak_allocated_vram_mib": (
        torch.cuda.max_memory_allocated() / 1024**2
    ),
    "peak_reserved_vram_mib": (
        torch.cuda.max_memory_reserved() / 1024**2
    ),
}

summary_path.write_text(
    json.dumps(summary, indent=2)
)

temporary_path = checkpoint_path.with_suffix(".pth.tmp")

torch.save(
    {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": NUM_STEPS,
        "seed": SEED,
        "classes": list(CLASSES),
        "summary": summary,
    },
    temporary_path,
)

os.replace(temporary_path, checkpoint_path)

assert checkpoint_path.is_file()
assert checkpoint_path.stat().st_size > 0

print("Metrics:", metrics_path)
print("Summary:", summary_path)
print("Checkpoint:", checkpoint_path)
print(
    "Checkpoint size: {:.1f} MiB".format(
        checkpoint_path.stat().st_size / 1024**2
    )
)

print("\n" + "=" * 65)
print("CARLA 20-STEP TRAINING PILOT PASSED")
print("=" * 65)
print("20 optimizer updates completed.")
print("Checkpoint saved.")
print("Full multi-epoch training has not started.")
