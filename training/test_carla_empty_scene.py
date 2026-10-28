#!/usr/bin/env python3
"""Test BEVFusion training loss with zero CARLA ground-truth boxes."""

import torch
from mmcv.parallel import DataContainer

from carla_dataset_config import build_dataset
from carla_model_loader import build_carla_model


SAMPLE_INDEX = 221


def unwrap(value):
    if isinstance(value, DataContainer):
        return value.data
    return value


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

    points = [
        gpu_tensor(sample["points"])
    ]

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

    boxes = unwrap(
        sample["gt_bboxes_3d"]
    ).to("cuda")

    labels = gpu_tensor(
        sample["gt_labels_3d"],
        dtype=torch.long,
    )

    metadata = unwrap(sample["metas"])

    return {
        "img": image,
        "points": points,
        **matrices,
        "metas": [metadata],
        "gt_bboxes_3d": [boxes],
        "gt_labels_3d": [labels],
    }


print("=" * 65)
print("STEP 11.5.9 — CARLA EMPTY-SCENE LOSS TEST")
print("=" * 65)

print("\n=== 1. LOAD EMPTY TRAINING SCENE ===")

dataset = build_dataset("train")

raw_annotations = dataset.get_ann_info(SAMPLE_INDEX)

assert len(raw_annotations["gt_labels_3d"]) == 0

sample = dataset[SAMPLE_INDEX]

assert sample is not None

inputs = prepare_sample(sample)

boxes = inputs["gt_bboxes_3d"][0]
labels = inputs["gt_labels_3d"][0]

print("Sample index:", SAMPLE_INDEX)
print("Number of boxes:", len(boxes))
print("Number of labels:", len(labels))
print("Box tensor shape:", tuple(boxes.tensor.shape))
print("Image tensor shape:", tuple(inputs["img"].shape))
print("LiDAR tensor shape:", tuple(inputs["points"][0].shape))

assert len(boxes) == 0
assert len(labels) == 0
assert inputs["img"].shape == (1, 1, 3, 256, 704)

print("\n=== 2. BUILD MODEL ===")

model, cfg = build_carla_model(
    pretrained=True,
    device="cuda",
)

model.train()

torch.cuda.reset_peak_memory_stats()

print("Model:", type(model).__name__)
print("Device:", next(model.parameters()).device)

print("\n=== 3. RUN EMPTY-SCENE TRAINING FORWARD ===")

with torch.no_grad():
    outputs = model(**inputs)

torch.cuda.synchronize()

assert isinstance(outputs, dict)

print("Returned entries:", len(outputs))

loss_count = 0

for name, value in outputs.items():
    values = (
        value if isinstance(value, (list, tuple))
        else [value]
    )

    for item in values:
        assert isinstance(item, torch.Tensor), name
        assert torch.isfinite(item).all(), (
            f"Non-finite value in {name}"
        )

        print(
            name,
            "=",
            float(item.detach().float().mean()),
        )

        if "loss" in name:
            loss_count += 1

assert loss_count >= 1, (
    "No loss-related outputs were produced"
)

print("\n=== 4. GPU MEMORY ===")

print(
    "Peak allocated VRAM: {:.1f} MiB".format(
        torch.cuda.max_memory_allocated() / 1024**2
    )
)

print("\n" + "=" * 65)
print("CARLA EMPTY-SCENE FORWARD TEST PASSED")
print("=" * 65)

print("Zero-box input accepted by the detection head.")
print("All returned values are finite.")
print("No backward pass was performed.")
print("No optimizer step was performed.")
print("No checkpoint was modified.")
