#!/usr/bin/env python3
"""Diagnostic: compare predicted and GT box centers in the LiDAR XY plane."""

from pathlib import Path

import torch
from mmcv.parallel import DataContainer

from carla_dataset_config import build_dataset, CLASSES
from carla_model_loader import build_carla_model

RUN = "heatmap_ab_20261028T091710Z"
CKPT = Path("/outputs/checkpoints") / RUN

SCENES = [1309, 228, 51]
RADII = [1.0, 2.0, 4.0]

MATRICES = [
    "camera2ego",
    "lidar2ego",
    "lidar2camera",
    "lidar2image",
    "camera_intrinsics",
    "camera2lidar",
    "img_aug_matrix",
    "lidar_aug_matrix",
]


def unwrap(value):
    if isinstance(value, DataContainer):
        return value.data
    return value


def gpu(value):
    return unwrap(value).to("cuda").contiguous()


def prepare(sample):
    result = {
        "img": gpu(sample["img"]).unsqueeze(0),
        "points": [gpu(sample["points"])],
        "metas": [unwrap(sample["metas"])],
    }

    for name in MATRICES:
        result[name] = gpu(sample[name]).unsqueeze(0)

    return result


def class_name(index):
    index = int(index)
    if 0 <= index < len(CLASSES):
        return CLASSES[index]
    return f"unknown_{index}"


print("=" * 66)
print("STEP 11.7.6 — PREDICTED BOX CENTER DIAGNOSTIC")
print("=" * 66)

dataset = build_dataset("val")
samples = [(i, dataset[i]) for i in SCENES]

model, _ = build_carla_model(
    pretrained=False,
    device="cpu",
)
model.cuda()

for variant in ["default", "prior_1percent"]:

    print(f"\n=== TRAINED VARIANT: {variant} ===")

    checkpoint = torch.load(
        CKPT / f"{variant}_step20.pth",
        map_location="cpu",
    )

    assert checkpoint["variant"] == variant
    assert checkpoint["step"] == 20

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )
    model.eval()

    total_gt = 0
    counts_any = {r: 0 for r in RADII}
    counts_class = {r: 0 for r in RADII}

    for index, sample in samples:
        with torch.no_grad():
            output = model(**prepare(sample))

        assert len(output) == 1
        prediction = output[0]

        pred_boxes = prediction["boxes_3d"].tensor.detach().cpu()
        pred_labels = prediction["labels_3d"].detach().cpu().long()
        pred_scores = prediction["scores_3d"].detach().cpu().float()

        gt_boxes = unwrap(sample["gt_bboxes_3d"]).tensor.detach().cpu()
        gt_labels = unwrap(sample["gt_labels_3d"]).detach().cpu().long()

        assert len(pred_boxes) == len(pred_labels) == len(pred_scores)
        assert len(gt_boxes) == len(gt_labels)
        assert len(pred_boxes) > 0
        assert len(gt_boxes) > 0

        assert torch.isfinite(pred_boxes).all()
        assert torch.isfinite(gt_boxes).all()

        # XY-only diagnostic: ignores box height, size,
        # orientation, overlap and confidence calibration.
        distances = torch.cdist(
            gt_boxes[:, :2].float(),
            pred_boxes[:, :2].float(),
        )

        total_gt += len(gt_boxes)

        print(
            f"\nScene {index:04d}: "
            f"{len(gt_boxes)} GT objects; "
            f"{len(pred_boxes)} proposals"
        )

        print(
            "  GT XY range:",
            gt_boxes[:, :2].min(0).values.tolist(),
            "to",
            gt_boxes[:, :2].max(0).values.tolist(),
        )

        print(
            "  Pred XY range:",
            pred_boxes[:, :2].min(0).values.tolist(),
            "to",
            pred_boxes[:, :2].max(0).values.tolist(),
        )

        for j in range(len(gt_boxes)):
            nearest_dist, nearest_index = distances[j].min(dim=0)
            nearest_index = int(nearest_index)

            same_class = pred_labels == gt_labels[j]

            if bool(same_class.any()):
                class_dist = distances[j, same_class].min().item()
            else:
                class_dist = float("inf")

            d = float(nearest_dist)

            for radius in RADII:
                if d <= radius:
                    counts_any[radius] += 1

                if class_dist <= radius:
                    counts_class[radius] += 1

            print(
                f"  GT {j:02d} "
                f"{class_name(gt_labels[j]):<12} | "
                f"nearest_XY={d:7.2f}m | "
                f"nearest_class="
                f"{class_name(pred_labels[nearest_index]):<12} | "
                f"score={pred_scores[nearest_index].item():.5f} | "
                f"same_class_XY={class_dist:7.2f}m"
            )

    print(f"\n=== SUMMARY: {variant} ===")
    print("GT objects examined:", total_gt)

    for radius in RADII:
        print(
            f"Within {radius:.0f}m XY | "
            f"any class: {counts_any[radius]}/{total_gt} | "
            f"same class: {counts_class[radius]}/{total_gt}"
        )

print("\n" + "=" * 66)
print("BOX CENTER DIAGNOSTIC COMPLETE")
print("=" * 66)
print("These are proximity diagnostics, not recall or mAP.")
print("No confidence threshold or one-to-one matching used.")
print("No training performed; checkpoints unchanged.")
