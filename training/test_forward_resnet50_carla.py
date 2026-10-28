#!/usr/bin/env python3

"""
Step 11.3.3 — First CARLA camera + LiDAR GPU inference test.

Tests:
- Real CARLA validation sample
- One-camera tensor batching
- LiDAR point-cloud batching
- Sensor calibration tensor dimensions
- Full ResNet50 BEVFusion inference
- Detection output structure
- GPU memory usage

No pretrained checkpoint is loaded.
No training is performed.
"""

import time
from pathlib import Path

import torch
import yaml

from mmcv import Config
from mmcv.parallel import DataContainer
from torchpack.utils.config import configs

from mmdet3d.utils import recursive_eval
from mmdet3d.models import build_model

from carla_dataset_config import build_dataset


BASE_CONFIG = Path(
    "/workspace/bevfusion/configs/nuscenes/det/"
    "transfusion/secfpn/camera+lidar/"
    "resnet50/convfuser.yaml"
)

OVERRIDE_CONFIG = Path(
    "/workspace/project/configs/one_camera/"
    "carla_resnet50_overrides.yaml"
)


def apply_overrides(target, overrides):
    for key, value in overrides.items():
        if key not in target:
            raise KeyError("Unknown configuration key: " + key)

        if isinstance(value, dict):
            apply_overrides(target[key], value)
        else:
            target[key] = value


def unpack(value):
    """Unwrap MMCV DataContainer without altering tensors."""
    if isinstance(value, DataContainer):
        return value.data
    return value


def to_cuda_tensor(value):
    value = unpack(value)

    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)

    return value.to(
        device="cuda",
        dtype=torch.float32,
    ).contiguous()


def gpu_memory_mb():
    return torch.cuda.memory_allocated() / (1024 ** 2)


def peak_gpu_memory_mb():
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


print("=" * 60)
print("STEP 11.3.3 — FIRST REAL CARLA BEVFUSION FORWARD PASS")
print("=" * 60)

assert torch.cuda.is_available(), "CUDA GPU is unavailable"

print("\n=== 1. GPU INFORMATION ===")

print("GPU:", torch.cuda.get_device_name(0))

properties = torch.cuda.get_device_properties(0)

print(
    "Total VRAM: {:.2f} GiB".format(
        properties.total_memory / (1024 ** 3)
    )
)


print("\n=== 2. LOAD MODEL CONFIGURATION ===")

configs.load(
    str(BASE_CONFIG),
    recursive=True,
)

with open(OVERRIDE_CONFIG, "r") as f:
    overrides = yaml.safe_load(f)

apply_overrides(configs, overrides)

cfg = Config(
    recursive_eval(configs),
    filename=str(BASE_CONFIG),
)

assert cfg.model.encoders.camera.backbone.type == "ResNet"
assert cfg.model.encoders.camera.backbone.depth == 50
assert cfg.model.encoders.camera.backbone.init_cfg is None
assert cfg.model.heads.object.num_classes == 6

print("Backbone: ResNet50")
print("Detection classes: 6")
print("Pretrained weights: disabled")


print("\n=== 3. LOAD REAL CARLA VALIDATION SAMPLE ===")

dataset = build_dataset("val")
sample = dataset[0]

print("Validation samples:", len(dataset))

image = to_cuda_tensor(sample["img"])
points = to_cuda_tensor(sample["points"])

assert image.shape == (1, 3, 256, 704)
assert points.ndim == 2
assert points.shape[1] == 5

# Add batch dimension:
# [N,C,H,W] -> [B,N,C,H,W]
image = image.unsqueeze(0)

# BEVFusion expects a list of point clouds,
# one tensor per batch item.
points = [points]

assert image.shape == (1, 1, 3, 256, 704)
assert len(points) == 1

print("Batched image:", tuple(image.shape))
print("LiDAR points:", tuple(points[0].shape))


print("\n=== 4. PREPARE SENSOR CALIBRATION ===")

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

matrices = {}

for key in matrix_keys:
    matrix = to_cuda_tensor(sample[key])

    # Each sample already contains camera dimensions
    # where required. Add only the batch dimension.
    matrix = matrix.unsqueeze(0)

    matrices[key] = matrix

    print(key, ":", tuple(matrix.shape))


print("\n=== 5. PREPARE METADATA ===")

metadata = unpack(sample["metas"])

assert isinstance(metadata, dict)

metas = [metadata]

print("Metadata batch length:", len(metas))
print("Metadata fields:", sorted(metadata.keys()))


print("\n=== 6. BUILD THE ACTUAL MODEL ===")

model = build_model(cfg.model)

model = model.cuda()
model.eval()

print("Model device:", next(model.parameters()).device)
print("Model GPU memory: {:.1f} MiB".format(
    gpu_memory_mb()
))

torch.cuda.reset_peak_memory_stats()


print("\n=== 7. RUN FULL CAMERA + LIDAR INFERENCE ===")

print("Running forward pass...")

torch.cuda.synchronize()
start = time.perf_counter()

with torch.no_grad():
    outputs = model(
        img=image,
        points=points,
        camera2ego=matrices["camera2ego"],
        lidar2ego=matrices["lidar2ego"],
        lidar2camera=matrices["lidar2camera"],
        lidar2image=matrices["lidar2image"],
        camera_intrinsics=matrices["camera_intrinsics"],
        camera2lidar=matrices["camera2lidar"],
        img_aug_matrix=matrices["img_aug_matrix"],
        lidar_aug_matrix=matrices["lidar_aug_matrix"],
        metas=metas,
    )

torch.cuda.synchronize()

elapsed = time.perf_counter() - start

print("Forward pass: COMPLETED")
print("Single-pass elapsed time: {:.3f} seconds".format(
    elapsed
))


print("\n=== 8. INSPECT DETECTION OUTPUTS ===")

assert isinstance(outputs, list)
assert len(outputs) == 1

result = outputs[0]

print("Output keys:", sorted(result.keys()))

for key in ["boxes_3d", "scores_3d", "labels_3d"]:
    assert key in result, f"Missing output: {key}"

boxes = result["boxes_3d"]
scores = result["scores_3d"]
labels = result["labels_3d"]

print("3D boxes:", tuple(boxes.tensor.shape))
print("Scores:", tuple(scores.shape))
print("Labels:", tuple(labels.shape))

assert len(boxes.tensor) == len(scores)
assert len(scores) == len(labels)

if len(labels) > 0:
    assert labels.min().item() >= 0
    assert labels.max().item() < 6

print("Output structure: PASSED")


print("\n=== 9. GPU MEMORY ===")

print(
    "Current allocated VRAM: {:.1f} MiB".format(
        gpu_memory_mb()
    )
)

print(
    "Peak allocated VRAM during forward: {:.1f} MiB".format(
        peak_gpu_memory_mb()
    )
)


print("\n" + "=" * 60)
print("FIRST CARLA RESNET50 FORWARD PASS PASSED")
print("=" * 60)

print("Camera count: 1")
print("LiDAR point features: 5")
print("Model classes: 6")
print("Prediction count:", len(scores))

print("\nIMPORTANT:")
print("Weights are randomly initialized.")
print("Predictions are not yet meaningful.")
print("Single-pass timing is not a reliable FPS benchmark.")
print("Training has not started.")
