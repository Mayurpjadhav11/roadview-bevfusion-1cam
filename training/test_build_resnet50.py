#!/usr/bin/env python3

"""
Step 11.2.5 — CARLA ResNet50 BEVFusion model construction.

This test:
- Loads NVIDIA's ResNet50 BEVFusion configuration.
- Applies CARLA six-class overrides.
- Resolves torchpack YAML expressions.
- Builds the full model on CPU.
- Checks the model architecture and parameters.

Does not download weights, run inference, or train.
"""

import resource
from pathlib import Path

import torch
import yaml

from mmcv import Config
from torchpack.utils.config import configs
from mmdet3d.utils import recursive_eval
from mmdet3d.models import build_model

from carla_dataset_config import (
    CLASSES,
    NUM_CLASSES,
    NUM_CAMERAS,
    IMAGE_SIZE,
    POINT_FEATURES,
)


# ============================================================
# 1. CONFIGURATION PATHS
# ============================================================

BASE_CONFIG = Path(
    "/workspace/bevfusion/configs/nuscenes/det/"
    "transfusion/secfpn/camera+lidar/"
    "resnet50/convfuser.yaml"
)

CARLA_OVERRIDE = Path(
    "/workspace/project/configs/one_camera/"
    "carla_resnet50_overrides.yaml"
)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def apply_overrides(target, overrides, prefix=""):
    """Apply CARLA overrides to the inherited configuration."""

    for key, value in overrides.items():

        full_key = (
            prefix + "." + key if prefix else key
        )

        if key not in target:
            raise KeyError(
                "Unknown configuration key: " + full_key
            )

        if isinstance(value, dict):
            apply_overrides(
                target[key],
                value,
                full_key,
            )
        else:
            target[key] = value


def memory_mb():
    """Return peak process RAM usage on Linux."""

    return (
        resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss / 1024.0
    )


def count_parameters(module):
    """Count PyTorch model parameters."""

    return sum(
        parameter.numel()
        for parameter in module.parameters()
    )


# ============================================================
# 3. ENVIRONMENT
# ============================================================

print("=" * 55)
print("STEP 11.2.5 — BUILD CARLA RESNET50 BEVFUSION")
print("=" * 55)

print("\n=== 1. ENVIRONMENT ===")

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Build device: CPU")
print("Initial peak RAM: {:.1f} MB".format(memory_mb()))

assert BASE_CONFIG.is_file()
assert CARLA_OVERRIDE.is_file()


# ============================================================
# 4. LOAD BASE CONFIGURATION
# ============================================================

print("\n=== 2. LOAD NVIDIA CONFIGURATION ===")

configs.load(
    str(BASE_CONFIG),
    recursive=True,
)

print("Base configuration: LOADED")


# ============================================================
# 5. APPLY CARLA OVERRIDES
# ============================================================

print("\n=== 3. APPLY CARLA OVERRIDES ===")

with open(CARLA_OVERRIDE, "r") as file:
    overrides = yaml.safe_load(file)

apply_overrides(configs, overrides)

print("CARLA class overrides: APPLIED")
print("ResNet50 pretrained initialization: DISABLED")


# ============================================================
# 6. RESOLVE ALL CONFIGURATION EXPRESSIONS
# ============================================================

print("\n=== 4. RESOLVE CONFIGURATION REFERENCES ===")

# IMPORTANT:
# recursive_eval resolves expressions such as:
#
#   ${image_size}
#   ${[image_size[0] // 8, image_size[1] // 8]}
#
# These must become numeric lists before model construction.

resolved_config = recursive_eval(configs)

cfg = Config(
    resolved_config,
    filename=str(BASE_CONFIG),
)

print("Configuration expressions: RESOLVED")

print("\nImage size:", cfg.image_size)

vtransform_cfg = cfg.model.encoders.camera.vtransform

print(
    "View transform image_size:",
    vtransform_cfg.image_size,
)

print(
    "View transform feature_size:",
    vtransform_cfg.feature_size,
)

print(
    "Voxel size:",
    cfg.voxel_size,
)

print(
    "Point-cloud range:",
    cfg.point_cloud_range,
)


# ============================================================
# 7. VALIDATE RESOLVED CONFIGURATION
# ============================================================

print("\n=== 5. CHECK RESOLVED CONFIGURATION ===")

assert list(cfg.image_size) == [256, 704]

assert list(vtransform_cfg.image_size) == [256, 704]

assert list(vtransform_cfg.feature_size) == [32, 88]

assert isinstance(
    vtransform_cfg.image_size,
    (list, tuple),
)

assert cfg.model.encoders.camera.backbone.type == "ResNet"

assert cfg.model.encoders.camera.backbone.depth == 50

assert cfg.model.encoders.camera.backbone.init_cfg is None

assert cfg.model.heads.object.num_classes == NUM_CLASSES

assert cfg.model.encoders.lidar.backbone.in_channels == (
    POINT_FEATURES
)

print("Numeric image dimensions: PASSED")
print("Numeric feature dimensions: PASSED")
print("ResNet50 architecture: PASSED")
print("Six-class detection head configuration: PASSED")


# ============================================================
# 8. CONSTRUCT THE ACTUAL MODEL
# ============================================================

print("\n=== 6. CONSTRUCT PYTORCH MODEL ===")

print("Building complete BEVFusion model...")

model = build_model(cfg.model)

model.cpu()
model.eval()

print("Model construction: PASSED")
print("Model class:", type(model).__name__)


# ============================================================
# 9. VERIFY CAMERA ENCODER
# ============================================================

print("\n=== 7. CAMERA ENCODER ===")

assert "camera" in model.encoders

camera = model.encoders["camera"]

for name in [
    "backbone",
    "neck",
    "vtransform",
]:
    assert name in camera

    component = camera[name]

    print(
        "{}: {} | parameters={:,}".format(
            name,
            type(component).__name__,
            count_parameters(component),
        )
    )


# ============================================================
# 10. VERIFY LIDAR ENCODER
# ============================================================

print("\n=== 8. LIDAR ENCODER ===")

assert "lidar" in model.encoders

lidar = model.encoders["lidar"]

for name in [
    "voxelize",
    "backbone",
]:
    assert name in lidar

    component = lidar[name]

    print(
        "{}: {} | parameters={:,}".format(
            name,
            type(component).__name__,
            count_parameters(component),
        )
    )


# ============================================================
# 11. VERIFY FUSION AND DECODER
# ============================================================

print("\n=== 9. BEV FUSION ===")

assert model.fuser is not None

print("Fusion:", type(model.fuser).__name__)

print(
    "Fusion parameters:",
    format(count_parameters(model.fuser), ","),
)


print("\n=== 10. BEV DECODER ===")

for name in [
    "backbone",
    "neck",
]:
    assert name in model.decoder

    component = model.decoder[name]

    print(
        "{}: {}".format(
            name,
            type(component).__name__,
        )
    )


# ============================================================
# 12. VERIFY DETECTION HEAD
# ============================================================

print("\n=== 11. DETECTION HEAD ===")

assert "object" in model.heads

head = model.heads["object"]

print("Detection head:", type(head).__name__)

actual_classes = getattr(
    head,
    "num_classes",
    None,
)

print("Detection class count:", actual_classes)

print("CARLA classes:", CLASSES)

assert actual_classes == 6

assert len(CLASSES) == 6


# ============================================================
# 13. PARAMETER AND DEVICE CHECK
# ============================================================

print("\n=== 12. MODEL PARAMETERS ===")

total_parameters = count_parameters(model)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    "Total parameters:",
    format(total_parameters, ","),
)

print(
    "Trainable parameters:",
    format(trainable_parameters, ","),
)

print(
    "Peak process RAM: {:.1f} MB".format(
        memory_mb()
    )
)

assert total_parameters > 0

assert all(
    parameter.device.type == "cpu"
    for parameter in model.parameters()
)

print("Model device: CPU")


# ============================================================
# 14. FINAL RESULT
# ============================================================

print("\n" + "=" * 55)
print("CARLA RESNET50 MODEL CONSTRUCTION PASSED")
print("=" * 55)

print("Architecture: BEVFusion")
print("Camera backbone: ResNet50")
print("Configured camera count:", NUM_CAMERAS)
print("Detection classes:", NUM_CLASSES)
print("Image dimensions:", IMAGE_SIZE)
print("Model parameters:", format(total_parameters, ","))

print("\nNo pretrained weights downloaded.")
print("No GPU forward pass performed.")
print("No training performed.")
