#!/usr/bin/env python3

"""Dataset helper for CARLA BEVFusion experiments."""

from carla_dataset_config import build_dataset


def build_carla_training_dataset():
    """
    Return the CARLA training dataset with empty
    ground-truth samples filtered during retrieval.

    Rejected samples are replaced using the original
    NuScenesDataset retry mechanism.
    """

    dataset = build_dataset("train")

    dataset.filter_empty_gt = True

    assert dataset.filter_empty_gt is True

    return dataset


def build_carla_validation_dataset():
    """
    Return validation data without training-time filtering.
    """

    dataset = build_dataset("val")

    assert dataset.filter_empty_gt is False

    return dataset


if __name__ == "__main__":

    import random
    import numpy as np

    from mmcv.parallel import DataContainer

    random.seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("STEP 11.5.11 — CARLA DATASET HELPER")
    print("=" * 60)

    train = build_carla_training_dataset()
    val = build_carla_validation_dataset()

    print("\n=== TRAINING DATASET ===")
    print("Samples:", len(train))
    print("Filter empty:", train.filter_empty_gt)

    print("\n=== VALIDATION DATASET ===")
    print("Samples:", len(val))
    print("Filter empty:", val.filter_empty_gt)

    assert len(train) == 16200
    assert len(val) == 1800

    assert train.filter_empty_gt is True
    assert val.filter_empty_gt is False

    print("\n=== TEST PREVIOUSLY EMPTY SAMPLE ===")

    raw = train.get_ann_info(221)

    print(
        "Original annotation count:",
        len(raw["gt_labels_3d"]),
    )

    assert len(raw["gt_labels_3d"]) == 0

    sample = train[221]

    assert sample is not None

    labels = sample["gt_labels_3d"]

    if isinstance(labels, DataContainer):
        labels = labels.data

    print("Returned label count:", len(labels))

    assert len(labels) > 0

    print("\n" + "=" * 60)
    print("CARLA DATASET HELPER PASSED")
    print("=" * 60)
    print("Training filtering: ENABLED")
    print("Validation filtering: UNCHANGED")
    print("No training performed.")
