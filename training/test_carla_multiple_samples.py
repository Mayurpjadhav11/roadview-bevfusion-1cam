#!/usr/bin/env python3

"""
BEVFusion CARLA multi-sample preprocessing test.

Reuses the previously validated preprocessing pipeline.
Checks selected training and validation samples.
Does not start model training.
"""

import runpy
import time

import torch

from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset


# ============================================================
# 1. REUSE EXISTING VALIDATED PIPELINE
# ============================================================

print("\n==========================================")
print("LOADING VALIDATED CARLA PIPELINE")
print("==========================================")

config = runpy.run_path(
    "/workspace/project/training/test_carla_pipeline.py"
)

train_dataset = config["dataset"]
pipeline = config["pipeline"]
classes = config["CLASSES"]

print("\nExisting pipeline loaded successfully.")


# ============================================================
# 2. INITIALIZE VALIDATION DATASET
# ============================================================

print("\n==========================================")
print("INITIALIZING VALIDATION DATASET")
print("==========================================")

val_dataset = NuScenesDataset(
    dataset_root="/dataset",
    ann_file=(
        "/outputs/experiments/carla_metadata/"
        "carla_infos_val.pkl"
    ),
    pipeline=pipeline,
    object_classes=classes,
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

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

assert len(train_dataset) == 16200
assert len(val_dataset) == 1800


# ============================================================
# 3. SELECT SAMPLES FROM DIFFERENT DATASET REGIONS
# ============================================================

train_indices = [
    0,
    1,
    10,
    100,
    500,
    1000,
    5000,
    8000,
    12000,
    16199,
]

val_indices = [
    0,
    1,
    10,
    100,
    500,
    1000,
    1799,
]


# ============================================================
# 4. VALIDATE A SINGLE SAMPLE
# ============================================================

def validate_sample(dataset, index):

    sample = dataset[index]

    assert sample is not None

    # Camera
    image = sample["img"].data

    assert tuple(image.shape) == (1, 3, 256, 704)
    assert torch.isfinite(image).all()

    # LiDAR
    points = sample["points"].data

    assert points.ndim == 2
    assert points.shape[1] == 5
    assert points.shape[0] > 0
    assert torch.isfinite(points).all()

    # 3D annotations
    boxes = sample["gt_bboxes_3d"].data.tensor
    labels = sample["gt_labels_3d"].data

    assert boxes.ndim == 2
    assert boxes.shape[1] == 9
    assert labels.ndim == 1
    assert len(boxes) == len(labels)
    assert torch.isfinite(boxes).all()

    if len(labels) > 0:
        assert bool((labels >= 0).all())
        assert bool((labels < len(classes)).all())

    # Calibration matrices
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

    for key in calibration_keys:
        assert key in sample, "Missing: " + key

        matrix = sample[key].data

        assert tuple(matrix.shape[-2:]) == (4, 4)
        assert torch.isfinite(matrix).all()

    assert sample["camera_intrinsics"].data.shape[0] == 1

    # Check camera-to-LiDAR inverse consistency
    camera_to_lidar = sample["camera2lidar"].data[0]
    lidar_to_camera = sample["lidar2camera"].data[0]

    identity = torch.eye(
        4,
        dtype=camera_to_lidar.dtype,
        device=camera_to_lidar.device,
    )

    inverse_error = torch.max(
        torch.abs(camera_to_lidar @ lidar_to_camera - identity)
    ).item()

    assert inverse_error < 1e-3, (
        "Camera/LiDAR inverse mismatch: "
        + str(inverse_error)
    )

    # Metadata
    metadata = sample["metas"].data
    token = str(metadata.get("token", "unknown"))

    return {
        "token": token,
        "points": int(points.shape[0]),
        "boxes": int(boxes.shape[0]),
        "inverse_error": inverse_error,
    }


# ============================================================
# 5. TEST DATASET SPLIT
# ============================================================

def test_split(name, dataset, indices):

    print("\n==========================================")
    print("TESTING", name)
    print("==========================================")

    passed = 0
    failed = 0
    empty_gt = 0

    for index in indices:

        start = time.perf_counter()

        try:
            result = validate_sample(dataset, index)

            elapsed = time.perf_counter() - start

            if result["boxes"] == 0:
                empty_gt += 1

            print(
                "PASS | index={:5d} | token={} | "
                "points={:6d} | boxes={:3d} | "
                "time={:.2f}s".format(
                    index,
                    result["token"],
                    result["points"],
                    result["boxes"],
                    elapsed,
                )
            )

            passed += 1

        except Exception as error:

            print(
                "FAIL | index={} | {}: {}".format(
                    index,
                    type(error).__name__,
                    error,
                )
            )

            failed += 1

    print("\n{} SUMMARY".format(name))
    print("Passed:", passed)
    print("Failed:", failed)
    print("Samples without retained boxes:", empty_gt)

    return passed, failed


# ============================================================
# 6. RUN TESTS
# ============================================================

train_passed, train_failed = test_split(
    "TRAINING DATASET",
    train_dataset,
    train_indices,
)

val_passed, val_failed = test_split(
    "VALIDATION DATASET",
    val_dataset,
    val_indices,
)


# ============================================================
# 7. FINAL SUMMARY
# ============================================================

print("\n==========================================")
print("FINAL MULTI-SAMPLE VALIDATION")
print("==========================================")

print("Training passed:", train_passed, "/", len(train_indices))
print("Validation passed:", val_passed, "/", len(val_indices))
print("Total failures:", train_failed + val_failed)

assert train_failed == 0
assert val_failed == 0

print("\nMULTI-SAMPLE PREPROCESSING TEST PASSED")
print("No model training was performed.")
