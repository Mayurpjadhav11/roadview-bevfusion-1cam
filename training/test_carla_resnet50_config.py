#!/usr/bin/env python3

"""
Validate the CARLA-specific BEVFusion ResNet50 model configuration.

This test:
1. Loads NVIDIA's ResNet50 BEVFusion configuration recursively.
2. Applies our CARLA project overrides.
3. Verifies the model architecture and class configuration.

No neural network is instantiated.
No weights are downloaded.
No training is performed.
"""

from pathlib import Path

import yaml
from torchpack.utils.config import configs

from carla_dataset_config import (
    CLASSES,
    NUM_CLASSES,
    NUM_CAMERAS,
    IMAGE_SIZE,
    POINT_FEATURES,
    POINT_CLOUD_RANGE,
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
# 2. APPLY NESTED CONFIGURATION OVERRIDES
# ============================================================

def apply_overrides(target, overrides, prefix=""):
    """
    Recursively apply changes to an existing configuration.

    Every overridden key must exist in the inherited
    configuration. This prevents silent mistakes caused
    by misspelled configuration keys.
    """

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
            print(
                "Updated {} = {}".format(
                    full_key,
                    value,
                )
            )


# ============================================================
# 3. LOAD NVIDIA CONFIGURATION
# ============================================================

print("\n==========================================")
print("STEP 11.2.4 — CARLA RESNET50 CONFIGURATION")
print("==========================================")

assert BASE_CONFIG.is_file(), (
    "Missing base configuration: " + str(BASE_CONFIG)
)

assert CARLA_OVERRIDE.is_file(), (
    "Missing CARLA overrides: " + str(CARLA_OVERRIDE)
)

print("\n=== LOADING NVIDIA BASE CONFIGURATION ===")

configs.load(
    str(BASE_CONFIG),
    recursive=True,
)

print("Base configuration loaded")


# ============================================================
# 4. LOAD AND APPLY CARLA OVERRIDES
# ============================================================

print("\n=== APPLYING CARLA OVERRIDES ===")

with open(CARLA_OVERRIDE, "r") as file:
    overrides = yaml.safe_load(file)

assert isinstance(overrides, dict)

apply_overrides(configs, overrides)


# ============================================================
# 5. VERIFY CAMERA ARCHITECTURE
# ============================================================

print("\n=== CAMERA ARCHITECTURE ===")

camera = configs.model.encoders.camera

print("Backbone:", camera.backbone.type)
print("Depth:", camera.backbone.depth)
print("Camera neck:", camera.neck.type)
print("View transform:", camera.vtransform.type)
print("Input image size:", list(configs.image_size))
print("Backbone initialization:", camera.backbone.init_cfg)

assert camera.backbone.type == "ResNet"
assert camera.backbone.depth == 50
assert camera.neck.type == "GeneralizedLSSFPN"
assert camera.vtransform.type == "DepthLSSTransform"
assert camera.backbone.init_cfg is None

assert list(configs.image_size) == IMAGE_SIZE
assert NUM_CAMERAS == 1


# ============================================================
# 6. VERIFY LIDAR ARCHITECTURE
# ============================================================

print("\n=== LIDAR ARCHITECTURE ===")

lidar = configs.model.encoders.lidar

print("Backbone:", lidar.backbone.type)
print("Input channels:", lidar.backbone.in_channels)
print("Voxel size:", list(configs.voxel_size))
print("Point-cloud range:", list(configs.point_cloud_range))

assert lidar.backbone.type == "SparseEncoder"
assert lidar.backbone.in_channels == POINT_FEATURES

assert list(configs.point_cloud_range) == (
    POINT_CLOUD_RANGE
)


# ============================================================
# 7. VERIFY FUSION
# ============================================================

print("\n=== FUSION ARCHITECTURE ===")

fuser = configs.model.fuser

print("Type:", fuser.type)
print("Input channels:", list(fuser.in_channels))
print("Output channels:", fuser.out_channels)

assert fuser.type == "ConvFuser"
assert list(fuser.in_channels) == [80, 256]
assert fuser.out_channels == 256


# ============================================================
# 8. VERIFY SIX-CLASS DETECTION HEAD
# ============================================================

print("\n=== DETECTION HEAD ===")

head = configs.model.heads.object

print("Head:", head.type)
print("Number of classes:", head.num_classes)
print("Classes:", list(configs.object_classes))

assert head.type == "TransFusionHead"
assert head.num_classes == 6

assert NUM_CLASSES == 6
assert list(configs.object_classes) == CLASSES

# Preserve the original detection regression outputs,
# including velocity.
assert head.bbox_coder.code_size == 10
assert list(head.common_heads.vel) == [2, 2]

print("Six-class detection configuration: PASSED")


# ============================================================
# 9. VERIFY INITIAL TEST BATCH SETTINGS
# ============================================================

print("\n=== INITIAL TEST SETTINGS ===")

print("Batch size:", configs.data.samples_per_gpu)
print("Data workers:", configs.data.workers_per_gpu)

assert configs.data.samples_per_gpu == 1
assert configs.data.workers_per_gpu == 1


# ============================================================
# 10. FINAL RESULT
# ============================================================

print("\n==========================================")
print("CARLA RESNET50 MODEL CONFIGURATION PASSED")
print("==========================================")

print("Camera backbone: ResNet50")
print("Configured cameras:", NUM_CAMERAS)
print("Detection classes:", NUM_CLASSES)
print("Image size:", IMAGE_SIZE)
print("LiDAR point features:", POINT_FEATURES)
print("Batch size:", configs.data.samples_per_gpu)

print("\nNo model was instantiated.")
print("No pretrained weights were downloaded.")
print("No training was performed.")
