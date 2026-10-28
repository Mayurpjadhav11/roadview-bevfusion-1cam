#!/usr/bin/env python3
"""Controlled A/B training: default vs 1% heatmap prior."""

import copy
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from mmcv.parallel import DataContainer

from carla_model_loader import build_carla_model
from carla_training_dataset import (
    build_carla_training_dataset,
    build_carla_validation_dataset,
)
from carla_heatmap_initialization import apply_heatmap_prior


SEED = 42
STEPS = 20
LR = 1e-5

RUN = datetime.now(timezone.utc).strftime(
    "heatmap_ab_%Y%m%dT%H%M%SZ"
)

OUT = Path("/outputs/experiments") / RUN
CKPT = Path("/outputs/checkpoints") / RUN


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


def prepare(sample):
    keys = [
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

    for key in keys:
        result[key] = gpu(sample[key]).unsqueeze(0)

    return result


def losses(outputs):
    result = {}

    for name, value in outputs.items():
        if "loss" not in name:
            continue

        values = value if isinstance(value, (tuple, list)) else [value]

        result[name.split("/")[-1]] = sum(
            item.mean() for item in values
        )

    expected = [
        "loss_heatmap",
        "layer_-1_loss_cls",
        "layer_-1_loss_bbox",
    ]

    assert all(name in result for name in expected)

    return result


def evaluate(model, samples, variant, stage):
    model.eval()

    # Ask only the root BEVFusion module to return
    # training losses; keep child BatchNorm in eval mode.
    model.training = True

    rows = []

    with torch.no_grad():
        for index, sample in samples:
            outputs = model(**prepare(sample))
            parts = losses(outputs)

            row = {
                "variant": variant,
                "stage": stage,
                "sample_index": index,
            }

            for name, tensor in parts.items():
                value = float(tensor.detach().cpu())
                assert np.isfinite(value)
                row[name] = value

            row["total"] = sum(
                row[name] for name in [
                    "loss_heatmap",
                    "layer_-1_loss_cls",
                    "layer_-1_loss_bbox",
                ]
            )

            rows.append(row)

    print(
        variant, stage,
        "| mean_heatmap:",
        round(np.mean([r["loss_heatmap"] for r in rows]), 4),
        "| mean_total:",
        round(np.mean([r["total"] for r in rows]), 4),
        flush=True,
    )

    return rows


print("=" * 65)
print("STEP 11.7.2 — CONTROLLED HEATMAP A/B EXPERIMENT")
print("=" * 65)

seed_all(SEED)
torch.backends.cudnn.benchmark = False

print("\n=== 1. CACHE IDENTICAL TRAINING INPUTS ===")

train = build_carla_training_dataset()
val = build_carla_validation_dataset()

assert train.filter_empty_gt is True
assert val.filter_empty_gt is False

indices = list(range(len(train)))
random.Random(SEED).shuffle(indices)

train_samples = []

for index in indices:
    if len(train.get_ann_info(index)["gt_labels_3d"]) == 0:
        continue

    sample = train[index]

    if len(unwrap(sample["gt_labels_3d"])) == 0:
        continue

    train_samples.append((index, sample))

    if len(train_samples) == STEPS:
        break

assert len(train_samples) == STEPS

val_samples = []

for index in range(len(val)):
    sample = val[index]

    if len(unwrap(sample["gt_labels_3d"])) > 0:
        val_samples.append((index, sample))

    if len(val_samples) == 6:
        break

assert len(val_samples) == 6

print("Training scenes:", [i for i, _ in train_samples])
print("Validation scenes:", [i for i, _ in val_samples])

print("\n=== 2. SAVE EXACT INITIAL WEIGHTS ===")

seed_all(SEED)

base, _ = build_carla_model(
    pretrained=True,
    device="cpu",
)

initial_state = {
    k: v.detach().cpu().clone()
    for k, v in base.state_dict().items()
}

del base

OUT.mkdir(parents=True, exist_ok=False)
CKPT.mkdir(parents=True, exist_ok=False)

torch.save(
    {"model": initial_state, "seed": SEED},
    CKPT / "exact_initial.pth",
)

print("Initial model saved.")

all_validation = []
all_training = []


for variant in ["default", "prior_1percent"]:

    print(f"\n=== 3. TRAIN VARIANT: {variant} ===")

    seed_all(SEED)

    model, _ = build_carla_model(
        pretrained=False,
        device="cpu",
    )

    model.load_state_dict(initial_state, strict=True)

    if variant == "prior_1percent":
        apply_heatmap_prior(model, 0.01)

    changed = [
        k for k, v in model.state_dict().items()
        if not torch.equal(v.cpu(), initial_state[k])
    ]

    expected = (
        []
        if variant == "default"
        else ["heads.object.heatmap_head.1.bias"]
    )

    assert changed == expected, changed

    model.cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.01,
    )

    # Evaluate both variants before training.
    all_validation.extend(
        evaluate(model, val_samples, variant, "before")
    )

    # Restore training mode for all child modules.
    model.train()

    # Reset RNG so stochastic training operations use
    # the same initial random sequence in both variants.
    seed_all(SEED + 100)

    for step, (index, sample) in enumerate(
        train_samples, start=1
    ):
        inputs = prepare(sample)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(**inputs)
        parts = losses(outputs)

        total = sum(parts.values())

        assert torch.isfinite(total)
        total.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 35.0
        )

        assert torch.isfinite(grad_norm)

        optimizer.step()

        row = {
            "variant": variant,
            "step": step,
            "sample_index": index,
            "total": float(total.detach().cpu()),
            "heatmap": float(
                parts["loss_heatmap"].detach().cpu()
            ),
            "classification": float(
                parts["layer_-1_loss_cls"].detach().cpu()
            ),
            "bbox": float(
                parts["layer_-1_loss_bbox"].detach().cpu()
            ),
        }

        all_training.append(row)

        print(
            f"{variant} | {step:02d}/{STEPS} | "
            f"loss={row['total']:.4f} | "
            f"heatmap={row['heatmap']:.4f}",
            flush=True,
        )

    all_validation.extend(
        evaluate(model, val_samples, variant, "after")
    )

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "variant": variant,
            "step": STEPS,
            "seed": SEED,
        },
        CKPT / f"{variant}_step20.pth",
    )

    del model, optimizer
    torch.cuda.empty_cache()


print("\n=== 4. SAVE COMPARISON RESULTS ===")

def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


write_csv(OUT / "training.csv", all_training)
write_csv(OUT / "validation.csv", all_validation)

(OUT / "settings.json").write_text(
    json.dumps(
        {
            "seed": SEED,
            "steps": STEPS,
            "learning_rate": LR,
            "training_indices": [i for i, _ in train_samples],
            "validation_indices": [i for i, _ in val_samples],
            "variants": ["default", "prior_1percent"],
            "note": (
                "Exact same initial state except optional "
                "heatmap bias; identical cached data."
            ),
        },
        indent=2,
    )
)

print("\n=== 5. VALIDATION SUMMARY ===")

for variant in ["default", "prior_1percent"]:
    for stage in ["before", "after"]:
        subset = [
            r for r in all_validation
            if r["variant"] == variant and r["stage"] == stage
        ]

        assert len(subset) == 6

        print(
            f"{variant:<16} {stage:<6} | "
            f"heatmap={np.mean([r['loss_heatmap'] for r in subset]):.4f} | "
            f"cls={np.mean([r['layer_-1_loss_cls'] for r in subset]):.4f} | "
            f"bbox={np.mean([r['layer_-1_loss_bbox'] for r in subset]):.4f} | "
            f"total={np.mean([r['total'] for r in subset]):.4f}"
        )

print("\nExperiments:", OUT)
print("Checkpoints:", CKPT)

print("\n" + "=" * 65)
print("CONTROLLED HEATMAP A/B EXPERIMENT COMPLETED")
print("=" * 65)
print("Both variants completed 20 optimizer updates.")
print("Original pilot checkpoint unchanged.")
print("Original NVIDIA source unchanged.")
