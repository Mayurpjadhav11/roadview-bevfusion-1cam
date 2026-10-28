#!/usr/bin/env python3
"""Read-only comparison of default and low-prior heatmap initialization."""

import math
import random
import numpy as np
import torch

from mmcv.parallel import DataContainer

from carla_dataset_config import build_dataset
from carla_model_loader import build_carla_model


SEED = 42
SCENES = [0, 1, 2, 3, 4, 5]
PRIOR = 0.01


def unwrap(value):
    return value.data if isinstance(value, DataContainer) else value


def gpu(value, dtype=torch.float32):
    value = unwrap(value)
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.to("cuda", dtype=dtype).contiguous()


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

    inputs = {
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
        inputs[key] = gpu(sample[key]).unsqueeze(0)

    return inputs


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("=" * 65)
print("STEP 11.6.8 — HEATMAP FOREGROUND PRIOR TEST")
print("=" * 65)

dataset = build_dataset("val")
samples = [(i, dataset[i]) for i in SCENES]

assert all(
    len(unwrap(sample["gt_labels_3d"])) > 0
    for _, sample in samples
)

print("\n=== 1. BUILD INITIAL MODEL ===")

model, cfg = build_carla_model(
    pretrained=True,
    device="cuda",
)

head = model.heads["object"]

biases = dict(head.heatmap_head.named_parameters())

assert "1.bias" in biases, (
    f"Unexpected heatmap head structure: {list(biases)}"
)

bias = biases["1.bias"]
assert bias.numel() == 6

original_bias = bias.detach().clone()
prior_logit = math.log(PRIOR / (1.0 - PRIOR))

print("Original biases:", original_bias.cpu().tolist())
print("Target prior:", PRIOR)
print("Target logit:", prior_logit)

model.eval()

# Only the root flag changes; all child modules stay in
# evaluation mode to keep BatchNorm/dropout deterministic.
model.training = True

captured = []

def capture(module, inputs, output):
    captured.append(output.detach())

handle = head.heatmap_head.register_forward_hook(capture)


def evaluate(name):
    print(f"\n=== {name} ===")

    probabilities = []
    losses = []

    for index, sample in samples:
        captured.clear()

        with torch.no_grad():
            outputs = model(**prepare(sample))

        assert len(captured) == 1

        probability = captured[0].float().sigmoid()

        loss_key = next(
            key for key in outputs
            if key.endswith("/loss_heatmap")
        )

        value = outputs[loss_key]
        values = value if isinstance(value, (list, tuple)) else [value]

        heatmap_loss = sum(
            float(v.detach().float().mean().cpu())
            for v in values
        )

        mean_probability = probability.mean().item()

        assert math.isfinite(heatmap_loss)
        assert math.isfinite(mean_probability)

        losses.append(heatmap_loss)
        probabilities.append(mean_probability)

        print(
            f"Scene {index:04d} | "
            f"mean_p={mean_probability:.7f} | "
            f"heatmap_loss={heatmap_loss:.4f}"
        )

    mean_p = sum(probabilities) / len(probabilities)
    mean_loss = sum(losses) / len(losses)

    print(f"Mean foreground probability: {mean_p:.7f}")
    print(f"Mean heatmap loss: {mean_loss:.4f}")

    return mean_p, mean_loss


try:
    baseline_p, baseline_loss = evaluate(
        "BASELINE INITIALIZATION"
    )

    print("\n=== 2. APPLY PRIOR IN MEMORY ===")

    with torch.no_grad():
        bias.fill_(prior_logit)

    assert torch.allclose(
        bias.detach(),
        torch.full_like(bias, prior_logit),
    )

    prior_p, prior_loss = evaluate(
        "ONE-PERCENT FOREGROUND PRIOR"
    )

finally:
    with torch.no_grad():
        bias.copy_(original_bias)

    handle.remove()

assert torch.equal(bias.detach(), original_bias)

print("\n=== 3. COMPARISON ===")

print("Original mean probability:", baseline_p)
print("Low-prior mean probability:", prior_p)

print("Original mean heatmap loss:", baseline_loss)
print("Low-prior mean heatmap loss:", prior_loss)

print(
    "Loss change (%):",
    100.0 * (prior_loss - baseline_loss) /
    max(abs(baseline_loss), 1e-12),
)

print("\n" + "=" * 65)
print("HEATMAP PRIOR COMPARISON COMPLETE")
print("=" * 65)
print("Original biases restored in memory.")
print("No training performed.")
print("No source or checkpoint modified.")
