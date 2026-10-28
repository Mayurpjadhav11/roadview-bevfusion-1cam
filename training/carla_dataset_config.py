"""
CARLA dataset configuration for NVIDIA BEVFusion.

Sensors:
    1 RGB camera (CAM_FRONT)
    1 LiDAR

Classes:
    car, truck, bus, motorcycle, bicycle, pedestrian

Initial preprocessing:
    Deterministic image augmentation
    No object pasting
    No map segmentation
    No historical LiDAR sweeps

This file does not modify NVIDIA's original source code.
"""

from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset


# ============================================================
# DATASET PATHS
# ============================================================

DATASET_ROOT = "/dataset"

METADATA_DIR = "/outputs/experiments/carla_metadata"

TRAIN_METADATA = (
    METADATA_DIR + "/carla_infos_train.pkl"
)

VAL_METADATA = (
    METADATA_DIR + "/carla_infos_val.pkl"
)


# ============================================================
# OBJECT CLASSES
# ============================================================

CLASSES = [
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "pedestrian",
]

NUM_CLASSES = len(CLASSES)


# ============================================================
# MODEL INPUT CONFIGURATION
# ============================================================

NUM_CAMERAS = 1

IMAGE_SIZE = [256, 704]

POINT_FEATURES = 5

POINT_CLOUD_RANGE = [
    -54.0, -54.0, -5.0,
     54.0,  54.0,  3.0,
]

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


# ============================================================
# SHARED PREPROCESSING PIPELINE
# ============================================================

def build_pipeline():
    """
    Build the deterministic CARLA preprocessing pipeline.

    Returns:
        List of pipeline component configurations.
    """

    return [

        dict(
            type="LoadMultiViewImageFromFiles",
            to_float32=True,
        ),

        dict(
            type="LoadPointsFromFile",
            coord_type="LIDAR",
            load_dim=POINT_FEATURES,
            use_dim=POINT_FEATURES,
            reduce_beams=None,
        ),

        dict(
            type="LoadPointsFromMultiSweeps",
            sweeps_num=0,
            load_dim=POINT_FEATURES,
            use_dim=POINT_FEATURES,
            pad_empty_sweeps=False,
            remove_close=False,
        ),

        dict(
            type="LoadAnnotations3D",
            with_bbox_3d=True,
            with_label_3d=True,
            with_attr_label=False,
        ),

        dict(
            type="ImageAug3D",
            final_dim=IMAGE_SIZE,
            resize_lim=[0.19, 0.19],
            bot_pct_lim=[0.0, 0.0],
            rot_lim=[0.0, 0.0],
            rand_flip=False,
            is_train=False,
        ),

        dict(
            type="GlobalRotScaleTrans",
            resize_lim=[1.0, 1.0],
            rot_lim=[0.0, 0.0],
            trans_lim=0.0,
            is_train=False,
        ),

        dict(
            type="PointsRangeFilter",
            point_cloud_range=POINT_CLOUD_RANGE,
        ),

        dict(
            type="ObjectRangeFilter",
            point_cloud_range=POINT_CLOUD_RANGE,
        ),

        dict(
            type="ObjectNameFilter",
            classes=CLASSES,
        ),

        dict(
            type="ImageNormalize",
            mean=IMAGE_MEAN,
            std=IMAGE_STD,
        ),

        dict(
            type="DefaultFormatBundle3D",
            classes=CLASSES,
        ),

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
# DATASET FACTORY
# ============================================================

def build_dataset(split="train"):
    """
    Create a CARLA dataset using the shared pipeline.

    Args:
        split:
            "train" or "val"

    Returns:
        Initialized NuScenesDataset.
    """

    if split == "train":
        metadata_file = TRAIN_METADATA

    elif split == "val":
        metadata_file = VAL_METADATA

    else:
        raise ValueError(
            "Unknown split: {}. Use train or val.".format(split)
        )

    dataset = NuScenesDataset(
        dataset_root=DATASET_ROOT,
        ann_file=metadata_file,
        pipeline=build_pipeline(),
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

    return dataset
