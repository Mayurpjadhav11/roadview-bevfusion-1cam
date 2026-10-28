#!/usr/bin/env python3

"""
CARLA one-camera + LiDAR preprocessing validation.

Uses the NVIDIA BEVFusion dataset pipeline.
Does not modify the original dataset or source code.
Does not start model training.
"""

import torch
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset


# ============================================================
# 1. DATASET CONFIGURATION
# ============================================================

CLASSES = [
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "pedestrian",
]

POINT_CLOUD_RANGE = [
    -54.0, -54.0, -5.0,
     54.0,  54.0,  3.0,
]

DATASET_ROOT = "/dataset"

TRAIN_METADATA = (
    "/outputs/experiments/carla_metadata/"
    "carla_infos_train.pkl"
)


# ============================================================
# 2. COMPLETE CARLA PREPROCESSING PIPELINE
# ============================================================

pipeline = [

    # Load RGB camera images
    dict(
        type="LoadMultiViewImageFromFiles",
        to_float32=True,
    ),

    # Load LiDAR points with five features
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        reduce_beams=None,
    ),

    # Use current LiDAR scan without repeated empty sweeps
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=0,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=False,
        remove_close=False,
    ),

    # Load 3D boxes and object labels
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
    ),

    # Resize and crop the single CARLA image
    dict(
        type="ImageAug3D",
        final_dim=[256, 704],
        resize_lim=[0.19, 0.19],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False,
    ),

    # Disable random LiDAR transformations for this test
    dict(
        type="GlobalRotScaleTrans",
        resize_lim=[1.0, 1.0],
        rot_lim=[0.0, 0.0],
        trans_lim=0.0,
        is_train=False,
    ),

    # Retain LiDAR points within the supported BEV range
    dict(
        type="PointsRangeFilter",
        point_cloud_range=POINT_CLOUD_RANGE,
    ),

    # Retain 3D object boxes within the supported range
    dict(
        type="ObjectRangeFilter",
        point_cloud_range=POINT_CLOUD_RANGE,
    ),

    # Keep the six CARLA detection classes
    dict(
        type="ObjectNameFilter",
        classes=CLASSES,
    ),

    # Normalize images for ResNet50
    dict(
        type="ImageNormalize",
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),

    # Format image, point cloud and annotation tensors
    dict(
        type="DefaultFormatBundle3D",
        classes=CLASSES,
    ),

    # Collect model inputs and calibration data
    dict(
        type="Collect3D",
        keys=[
            "img",
            "points",
            "gt_bboxes_3d",
            "gt_labels_3d",
        ],
        meta_keys=[
            "camera_intrinsics",
            "camera2ego",
            "lidar2ego",
            "lidar2camera",
            "camera2lidar",
            "lidar2image",
            "img_aug_matrix",
            "lidar_aug_matrix",
        ],
    ),
]


# ============================================================
# 3. INITIALIZE DATASET
# ============================================================

print("\n========== INITIALIZING DATASET ==========")

dataset = NuScenesDataset(
    dataset_root=DATASET_ROOT,
    ann_file=TRAIN_METADATA,
    pipeline=pipeline,
    object_classes=CLASSES,
    modality=dict(
        use_camera=True,
        use_lidar=True,
        use_radar=False,
        use_map=False,
        use_external=False,
    ),
    with_velocity=True,
    use_valid_flag=True,
    filter_empty_gt=False,
    test_mode=False,
)

print("Dataset size:", len(dataset))

assert len(dataset) == 16200, "Unexpected dataset size"


# ============================================================
# 4. LOAD ONE TRAINING SAMPLE
# ============================================================

print("\n========== LOADING SAMPLE ==========")

sample = dataset[0]

assert sample is not None, "Sample loading failed"

print("Sample loaded successfully")

print("\nCollected keys:")
for key in sample:
    print(" -", key)


# ============================================================
# 5. CAMERA VALIDATION
# ============================================================

print("\n========== CAMERA ==========")

image = sample["img"].data

print("Image shape:", tuple(image.shape))
print("Image dtype:", image.dtype)
print("Image min:", float(image.min()))
print("Image max:", float(image.max()))

assert tuple(image.shape) == (1, 3, 256, 704)
assert torch.isfinite(image).all()

print("Camera validation: PASSED")


# ============================================================
# 6. LIDAR VALIDATION
# ============================================================

print("\n========== LIDAR ==========")

points = sample["points"].data

print("Point cloud shape:", tuple(points.shape))
print("Point dtype:", points.dtype)
print("First point:", points[0].tolist())

assert points.ndim == 2
assert points.shape[1] == 5
assert points.shape[0] > 0
assert torch.isfinite(points).all()

print("LiDAR validation: PASSED")


# ============================================================
# 7. 3D ANNOTATION VALIDATION
# ============================================================

print("\n========== 3D ANNOTATIONS ==========")

boxes = sample["gt_bboxes_3d"].data.tensor
labels = sample["gt_labels_3d"].data

print("Boxes shape:", tuple(boxes.shape))
print("Labels shape:", tuple(labels.shape))
print("Number of retained objects:", len(boxes))

assert boxes.ndim == 2
assert boxes.shape[1] == 9
assert labels.ndim == 1
assert len(boxes) == len(labels)
assert torch.isfinite(boxes).all()

if len(boxes):
    assert bool((labels >= 0).all())
    assert bool((labels < len(CLASSES)).all())

    print("\nObjects in this sample:")
    for index, label in enumerate(labels.tolist()):
        print(
            "  Object {}: {}".format(
                index + 1,
                CLASSES[label],
            )
        )

    print("\nFirst 3D box:")
    print(boxes[0].tolist())
else:
    print("This sample contains no retained objects.")

print("Annotation validation: PASSED")


# ============================================================
# 8. CAMERA AND LIDAR CALIBRATION VALIDATION
# ============================================================

print("\n========== CALIBRATION ==========")

calibration_keys = [
    "camera_intrinsics",
    "camera2ego",
    "lidar2ego",
    "lidar2camera",
    "camera2lidar",
    "lidar2image",
    "img_aug_matrix",
    "lidar_aug_matrix",
]

calibration = {}

for key in calibration_keys:

    assert key in sample, "Missing calibration: " + key

    tensor = sample[key].data
    calibration[key] = tensor

    print("{}: {}".format(key, tuple(tensor.shape)))

    assert torch.isfinite(tensor).all()

    assert tuple(tensor.shape[-2:]) == (4, 4), (
        "Unexpected matrix shape for " + key
    )

intrinsics = calibration["camera_intrinsics"]

assert intrinsics.ndim == 3
assert intrinsics.shape[0] == 1

for key in [
    "camera2ego",
    "lidar2camera",
    "camera2lidar",
    "lidar2image",
    "img_aug_matrix",
]:
    assert calibration[key].shape[0] == 1, (
        "Expected one camera for " + key
    )

print("\nCalibrated cameras:", intrinsics.shape[0])

print("\nCamera intrinsic matrix:")
print(intrinsics[0])

print("\nImage augmentation matrix:")
print(calibration["img_aug_matrix"][0])

print("\nLiDAR augmentation matrix:")
print(calibration["lidar_aug_matrix"])

print("\nCalibration validation: PASSED")


# ============================================================
# 9. GENERAL METADATA
# ============================================================

print("\n========== METADATA ==========")

metadata = sample["metas"].data

print("Available metadata fields:")
for key in metadata:
    print(" -", key)

if "token" in metadata:
    print("\nSample token:", metadata["token"])

if "lidar_path" in metadata:
    print("LiDAR file:", metadata["lidar_path"])


# ============================================================
# 10. FINAL RESULT
# ============================================================

print("\n==========================================")
print("FULL CARLA PREPROCESSING PIPELINE PASSED")
print("==========================================")

print("Training samples:", len(dataset))
print("Number of cameras:", intrinsics.shape[0])
print("Camera tensor:", tuple(image.shape))
print("LiDAR points:", tuple(points.shape))
print("3D boxes:", tuple(boxes.shape))
print("Object labels:", tuple(labels.shape))

print("\nAll checks passed for sample 0.")
print("No model training was performed.")
