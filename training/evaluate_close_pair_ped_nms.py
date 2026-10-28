from pathlib import Path
from unittest import mock
import types
import csv
import numpy as np
import torch
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
from mmcv.parallel import DataContainer

from carla_model_loader import build_carla_model
from carla_training_dataset import build_carla_training_dataset
from carla_dataset_config import CLASSES

BASE = Path(
    "/outputs/checkpoints/"
    "close_pair_ped_nms_20261028T112941Z"
)

INITIAL = Path(
    "/outputs/checkpoints/"
    "class_conditioned_dim_120_20261028T111607Z/"
    "step120.pth"
)

STEPS = list(range(10, 81, 10))
PED = list(CLASSES).index("pedestrian")
CAR = list(CLASSES).index("car")

MATRICES = [
    "camera2ego", "lidar2ego", "lidar2camera",
    "lidar2image", "camera_intrinsics",
    "camera2lidar", "img_aug_matrix",
    "lidar_aug_matrix",
]

def unwrap(x):
    return x.data if isinstance(x, DataContainer) else x

def gpu(x):
    x = unwrap(x)
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return x.to("cuda", dtype=torch.float32).contiguous()

def polygon(corners):
    xy = np.unique(corners[:, :2], axis=0)
    if len(xy) < 3:
        return np.empty((0, 2))
    return xy[ConvexHull(xy).vertices]

def cross(a, b):
    return float(a[0]*b[1] - a[1]*b[0])

def area(p):
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return abs(float(
        np.dot(x, np.roll(y, -1))
        - np.dot(y, np.roll(x, -1))
    )) / 2.0

def clip(subject, boundary):
    result = list(subject)

    for i in range(len(boundary)):
        if not result:
            break

        a = boundary[i]
        b = boundary[(i+1) % len(boundary)]
        previous = result
        result = []

        def inside(p):
            return cross(b-a, p-a) >= -1e-9

        def intersection(s, e):
            d = e-s
            edge = b-a
            den = cross(d, edge)
            if abs(den) < 1e-12:
                return e
            return s + d * cross(a-s, edge)/den

        for j, e in enumerate(previous):
            s = previous[j-1]
            if inside(e):
                if not inside(s):
                    result.append(intersection(s, e))
                result.append(e)
            elif inside(s):
                result.append(intersection(s, e))

    return np.asarray(result).reshape(-1, 2)

def iou3d(a, b):
    pa, pb = polygon(a), polygon(b)
    aa, ab = area(pa), area(pb)

    if aa <= 0 or ab <= 0:
        return 0.0

    intersection_area = area(clip(pa, pb))

    za0, za1 = a[:, 2].min(), a[:, 2].max()
    zb0, zb1 = b[:, 2].min(), b[:, 2].max()

    ha = max(0.0, za1-za0)
    hb = max(0.0, zb1-zb0)
    hi = max(0.0, min(za1, zb1)-max(za0, zb0))

    intersection = intersection_area * hi
    union = aa*ha + ab*hb - intersection

    return float(intersection / max(union, 1e-12))

print("=== STEP 11.8.50: CHECKPOINT EVALUATION ===",
      flush=True)

for step in STEPS:
    assert (BASE / f"step{step}.pth").is_file()

assert INITIAL.is_file()

saved = torch.load(INITIAL, map_location="cpu")
index = int(saved["sample_index"])

dataset = build_carla_training_dataset()
sample = dataset[index]

gt_obj = unwrap(sample["gt_bboxes_3d"])
gt_corners = gt_obj.corners.cpu().numpy()
gt_labels = torch.as_tensor(
    unwrap(sample["gt_labels_3d"])
).cpu().numpy()

assert len(gt_corners) == 5
assert int((gt_labels == CAR).sum()) == 1
assert int((gt_labels == PED).sum()) == 4

inputs = {
    "img": gpu(sample["img"]).unsqueeze(0),
    "points": [gpu(sample["points"])],
    "metas": [unwrap(sample["metas"])],
}

for key in MATRICES:
    inputs[key] = gpu(sample[key]).unsqueeze(0)

torch.manual_seed(42)

model, _ = build_carla_model(
    pretrained=False,
    device="cpu",
)

head = model.heads["object"]

assert head.nms_kernel_size == 3
assert head.num_decoder_layers == 1
assert head.num_classes == len(CLASSES)

head.class_dim_offset = torch.nn.Embedding(
    head.num_classes, 3
)

model.cuda().eval()

def dimension_hook(module, args, result):
    corrections = head.class_dim_offset(
        head.query_labels
    ).transpose(1, 2)

    updated = dict(result)
    updated["dim"] = result["dim"] + corrections
    return updated

hook = (
    head.prediction_heads[0]
    .register_forward_hook(dimension_hook)
)

original_pool = F.max_pool2d
original_forward = head.forward_single
hits = {"count": 0}

def pedestrian_pool(x, kernel_size, *args, **kwargs):
    result = original_pool(
        x, kernel_size, *args, **kwargs
    )

    if (
        x.ndim == 4
        and x.shape[1] == head.num_classes
        and kernel_size == 3
    ):
        hits["count"] += 1
        result = result.clone()
        result[:, PED, :, :] = x[:, PED, 1:-1, 1:-1]

    return result

def patched_forward(self, features, image_features, metas):
    with mock.patch.object(
        F, "max_pool2d", new=pedestrian_pool
    ):
        return original_forward(
            features, image_features, metas
        )

head.forward_single = types.MethodType(
    patched_forward, head
)

def evaluate(name, path):
    checkpoint = torch.load(path, map_location="cpu")

    assert int(checkpoint["sample_index"]) == index

    model.load_state_dict(
        checkpoint["model"],
        strict=True
    )

    model.eval()
    hits["count"] = 0

    with torch.no_grad():
        prediction = model(**inputs)[0]

    assert hits["count"] > 0

    boxes = prediction["boxes_3d"]
    corners = boxes.corners.detach().cpu().numpy()
    labels = prediction["labels_3d"].detach().cpu().numpy()
    scores = prediction["scores_3d"].detach().cpu().numpy()

    ng = len(gt_corners)
    npred = len(corners)

    cost = np.full((ng, npred+ng), 2.0)
    overlaps = np.zeros((ng, npred))

    for g in range(ng):
        cost[g, npred+g] = 1.0

        for p in range(npred):
            if gt_labels[g] != labels[p]:
                continue

            value = iou3d(gt_corners[g], corners[p])

            overlaps[g, p] = value
            cost[g, p] = 1.0-value

    rows, cols = linear_sum_assignment(cost)

    per_gt = [0.0] * ng
    matched = {}

    for g, p in zip(rows, cols):
        if p < npred and overlaps[g, p] > 0:
            per_gt[int(g)] = float(overlaps[g, p])
            matched[int(g)] = int(p)

    count25 = sum(v >= 0.25 for v in per_gt)
    count50 = sum(v >= 0.50 for v in per_gt)
    mean = float(np.mean(per_gt))
    car_iou = per_gt[0]

    assert int(gt_labels[0]) == CAR

    passed = (
        car_iou >= 0.50
        and count25 == 5
    )

    print(f"\n=== {name} ===", flush=True)
    print("Mean 3D IoU:", round(mean, 5))
    print("Objects IoU >= 0.25:", f"{count25}/5")
    print("Objects IoU >= 0.50:", f"{count50}/5")
    print("Car IoU:", round(car_iou, 5))
    print(
        "Per-GT IoU:",
        np.round(per_gt, 4).tolist()
    )

    for g in range(ng):
        p = matched.get(g)
        if p is not None:
            print(
                f"GT{g} {CLASSES[int(gt_labels[g])]} "
                f"pred={p} "
                f"IoU={per_gt[g]:.4f} "
                f"score={scores[p]:.5f}"
            )
        else:
            print(f"GT{g}: unmatched")

    print("ACCEPTANCE:", "PASS" if passed else "FAIL")

    return {
        "name": name,
        "mean_iou": mean,
        "count25": count25,
        "count50": count50,
        "car_iou": car_iou,
        "gt0": per_gt[0],
        "gt1": per_gt[1],
        "gt2": per_gt[2],
        "gt3": per_gt[3],
        "gt4": per_gt[4],
        "passed": passed,
    }

results = []

# Compare the source checkpoint under the
# same pedestrian-specific NMS policy.
results.append(
    evaluate("SOURCE_120", INITIAL)
)

for step in STEPS:
    results.append(
        evaluate(
            f"FINETUNE_{step}",
            BASE / f"step{step}.pth"
        )
    )

head.forward_single = original_forward
hook.remove()

print("\n=== STEP 11.8.50 FINAL RANKING ===")

for row in sorted(
    results,
    key=lambda r: (
        r["count25"],
        r["mean_iou"]
    ),
    reverse=True
):
    print(
        f"{row['name']:<16} "
        f"count25={row['count25']}/5 "
        f"meanIoU={row['mean_iou']:.4f} "
        f"car={row['car_iou']:.4f} "
        f"GT2={row['gt2']:.4f} "
        f"GT4={row['gt4']:.4f} "
        f"acceptance={'PASS' if row['passed'] else 'FAIL'}"
    )

output_csv = Path(
    "/outputs/experiments/"
    "close_pair_ped_nms_evaluation.csv"
)

with output_csv.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(results[0].keys())
    )
    writer.writeheader()
    writer.writerows(results)

passing = [
    r["name"] for r in results if r["passed"]
]

print("\n=== ACCEPTANCE CHECK ===")
print("Passing checkpoints:", passing)
print("Results CSV:", output_csv)

print("\nSTEP 11.8.50 COMPLETE")
print("Evaluation only; no training.")
print("Original BEVFusion source unchanged.")
