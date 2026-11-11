import ast
import csv
import random
import types
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
from mmcv.parallel import DataContainer

from carla_dataset_config import build_dataset, CLASSES
from carla_model_loader import build_carla_model

SEED = 119
NUM_SCENES = 12
IOU_THRESHOLD = 0.25
SCORE_THRESHOLDS = [0.0, 0.005, 0.01, 0.05, 0.10]

PED = list(CLASSES).index("pedestrian")
CAR = list(CLASSES).index("car")

ROOT = Path("/outputs/checkpoints")
SOURCE = ROOT / (
    "hard_ped_center_fixed_20261028T114303Z/step40.pth"
)
PILOT = ROOT / "carla_multiscene_20_20261028T133530Z"

CHECKPOINTS = [
    ("SOURCE_118", SOURCE),
    ("MULTISCENE_10", PILOT / "step10.pth"),
    ("MULTISCENE_20", PILOT / "step20.pth"),
]

REPORT = Path(
    "/outputs/experiments/"
    "carla_multiscene_detection_validation_12.csv"
)

# Import ONLY the previously verified 3D-IoU helpers.
# This does not execute the original evaluator.
verified_evaluator = Path(
    "/workspace/project/training/"
    "evaluate_hard_ped_center_fixed.py"
)

required = {"polygon", "cross", "area", "clip", "iou3d"}
tree = ast.parse(verified_evaluator.read_text())

functions = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name in required
]

assert {node.name for node in functions} == required

module = ast.Module(body=functions, type_ignores=[])
exec(
    compile(module, str(verified_evaluator), "exec"),
    globals()
)

def unwrap(x):
    return x.data if isinstance(x, DataContainer) else x

def gpu(x):
    x = unwrap(x)
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return x.to("cuda", dtype=torch.float32).contiguous()

MATRICES = [
    "camera2ego", "lidar2ego", "lidar2camera",
    "lidar2image", "camera_intrinsics",
    "camera2lidar", "img_aug_matrix",
    "lidar_aug_matrix",
]

def prepare_inference(sample):
    # Deliberately NO gt_bboxes_3d or gt_labels_3d.
    data = {
        "img": gpu(sample["img"]).unsqueeze(0),
        "points": [gpu(sample["points"])],
        "metas": [unwrap(sample["metas"])],
    }
    for key in MATRICES:
        data[key] = gpu(sample[key]).unsqueeze(0)
    return data

def box_geometry(corners):
    if len(corners) == 0:
        return np.zeros((0, 2)), np.zeros(0)

    centers = corners[:, :, :2].mean(axis=1)
    radii = np.linalg.norm(
        corners[:, :, :2] - centers[:, None, :],
        axis=2
    ).max(axis=1)
    return centers, radii

def overlap_matrix(gt_corners, gt_labels,
                   pred_corners, pred_labels):
    matrix = np.zeros(
        (len(gt_corners), len(pred_corners)),
        dtype=np.float32
    )

    gc, gr = box_geometry(gt_corners)
    pc, pr = box_geometry(pred_corners)

    for g in range(len(gt_corners)):
        for p in np.flatnonzero(pred_labels == gt_labels[g]):
            # Cheap geometric rejection before polygon clipping.
            distance = np.linalg.norm(gc[g] - pc[p])
            if distance > gr[g] + pr[p] + 0.01:
                continue

            if (
                gt_corners[g, :, 2].max()
                <= pred_corners[p, :, 2].min()
            ):
                continue

            if (
                pred_corners[p, :, 2].max()
                <= gt_corners[g, :, 2].min()
            ):
                continue

            matrix[g, p] = iou3d(
                gt_corners[g], pred_corners[p]
            )

    return matrix

def count_detections(gt_labels, pred_labels,
                     scores, overlaps, score_threshold,
                     iou_threshold):
    nclass = len(CLASSES)
    tp = np.zeros(nclass, dtype=int)
    fp = np.zeros(nclass, dtype=int)
    fn = np.bincount(
        gt_labels.astype(int), minlength=nclass
    ).astype(int)

    used_gt = set()

    selected = np.flatnonzero(
        scores >= score_threshold
    )

    # Score-ranked greedy matching, as used for
    # confidence-based object detection evaluation.
    selected = selected[
        np.argsort(-scores[selected], kind="stable")
    ]

    for p in selected:
        cls = int(pred_labels[p])

        eligible = [
            g for g in range(len(gt_labels))
            if g not in used_gt
            and int(gt_labels[g]) == cls
            and overlaps[g, p] >= iou_threshold
        ]

        if not eligible:
            fp[cls] += 1
            continue

        g = max(
            eligible,
            key=lambda candidate: overlaps[candidate, p]
        )

        used_gt.add(g)
        tp[cls] += 1
        fn[cls] -= 1

    return tp, fp, fn

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False

print("=== STEP 11.9.4: DETECTION VALIDATION ===", flush=True)

dataset = build_dataset("val")
assert len(dataset) == 1800
assert dataset.filter_empty_gt is False

candidates = list(range(len(dataset)))
random.Random(SEED).shuffle(candidates)

indices = []
prefixes = set()

for index in candidates:
    prefix = str(
        dataset.data_infos[index]["token"]
    ).rsplit("_", 1)[0]

    if prefix in prefixes:
        continue

    indices.append(index)
    prefixes.add(prefix)

    if len(indices) == NUM_SCENES:
        break

assert len(indices) == NUM_SCENES
assert len(prefixes) == NUM_SCENES

print("Validation indices:", indices, flush=True)
print("Scene prefixes:", sorted(prefixes), flush=True)

for _, path in CHECKPOINTS:
    assert path.is_file(), path

model, _ = build_carla_model(
    pretrained=False,
    device="cpu"
)

head = model.heads["object"]
assert head.nms_kernel_size == 3
assert head.num_classes == len(CLASSES)
assert head.num_decoder_layers == 1

head.class_dim_offset = torch.nn.Embedding(
    head.num_classes, 3
)

model.cuda().eval()

def dimension_hook(module, args, output):
    result = dict(output)
    result["dim"] = output["dim"] + (
        head.class_dim_offset(head.query_labels)
        .transpose(1, 2)
    )
    return result

handle = head.prediction_heads[0].register_forward_hook(
    dimension_hook
)

original_pool = F.max_pool2d
original_forward = head.forward_single
nms_calls = {"count": 0}

def pedestrian_pool(x, kernel_size, *args, **kwargs):
    pooled = original_pool(x, kernel_size, *args, **kwargs)

    if (
        x.ndim == 4
        and x.shape[1] == head.num_classes
        and kernel_size == 3
    ):
        nms_calls["count"] += 1
        pooled = pooled.clone()
        pooled[:, PED] = x[:, PED, 1:-1, 1:-1]

    return pooled

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

results = []
summaries = {}

for model_name, checkpoint_path in CHECKPOINTS:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu"
    )
    model.load_state_dict(
        checkpoint["model"], strict=True
    )
    model.eval()

    totals = {
        threshold: [
            np.zeros(len(CLASSES), dtype=int)
            for _ in range(3)
        ]
        for threshold in SCORE_THRESHOLDS
    }

    car50 = [
        np.zeros(len(CLASSES), dtype=int)
        for _ in range(3)
    ]

    print(f"\nMODEL: {model_name}", flush=True)

    for iteration, index in enumerate(indices, 1):
        sample = dataset[index]
        assert sample is not None

        gt_boxes = unwrap(sample["gt_bboxes_3d"])
        gt_labels = torch.as_tensor(
            unwrap(sample["gt_labels_3d"])
        ).cpu().numpy().astype(int)
        if len(gt_boxes) == 0:
            # Empty scenes are valid during evaluation.
            # Keep them so predicted boxes count as FP.
            gt_corners = np.empty(
                (0, 8, 3), dtype=np.float32
            )
        else:
            gt_corners = gt_boxes.corners.cpu().numpy()

        inputs = prepare_inference(sample)

        nms_calls["count"] = 0
        with torch.no_grad():
            prediction = model(**inputs)[0]

        assert nms_calls["count"] > 0

        pred_corners = (
            prediction["boxes_3d"]
            .corners.detach().cpu().numpy()
        )
        pred_labels = (
            prediction["labels_3d"]
            .detach().cpu().numpy().astype(int)
        )
        pred_scores = (
            prediction["scores_3d"]
            .detach().cpu().numpy()
        )

        assert np.isfinite(pred_scores).all()
        assert np.isfinite(pred_corners).all()

        overlaps = overlap_matrix(
            gt_corners, gt_labels,
            pred_corners, pred_labels
        )

        for threshold in SCORE_THRESHOLDS:
            values = count_detections(
                gt_labels, pred_labels,
                pred_scores, overlaps,
                threshold, IOU_THRESHOLD
            )
            for total, value in zip(totals[threshold], values):
                total += value

        values50 = count_detections(
            gt_labels, pred_labels,
            pred_scores, overlaps,
            0.01, 0.50
        )
        for total, value in zip(car50, values50):
            total += value

        token = dataset.data_infos[index]["token"]
        print(
            f"  {iteration:02d}/{NUM_SCENES} "
            f"index={index} token={token} "
            f"GT={len(gt_labels)} "
            f"pred={len(pred_labels)}",
            flush=True
        )

        del inputs, prediction, overlaps

    summaries[model_name] = totals

    print(f"\n=== {model_name} SUMMARY ===", flush=True)
    print("IoU threshold: 0.25")

    for threshold in SCORE_THRESHOLDS:
        tp, fp, fn = totals[threshold]
        T, FP, N = tp.sum(), fp.sum(), fn.sum()

        precision = T / (T + FP) if T + FP else 0.0
        recall = T / (T + N) if T + N else 0.0

        print(
            f"score>={threshold:.3f} "
            f"TP={T} FP={FP} FN={N} "
            f"precision={precision:.4f} "
            f"recall={recall:.4f}",
            flush=True
        )

        for cls, name in enumerate(CLASSES):
            results.append({
                "model": model_name,
                "score_threshold": threshold,
                "iou_threshold": IOU_THRESHOLD,
                "class": name,
                "tp": int(tp[cls]),
                "fp": int(fp[cls]),
                "fn": int(fn[cls]),
            })

    print("Class breakdown at score>=0.01:")

    tp, fp, fn = totals[0.01]

    for cls, name in enumerate(CLASSES):
        print(
            f"  {name:<12} "
            f"TP={tp[cls]} FP={fp[cls]} FN={fn[cls]}",
            flush=True
        )

    tp50, fp50, fn50 = car50
    print(
        f"Car IoU>=0.50 score>=0.01: "
        f"TP={tp50[CAR]} "
        f"FP={fp50[CAR]} "
        f"FN={fn50[CAR]}",
        flush=True
    )

REPORT.parent.mkdir(parents=True, exist_ok=True)

with REPORT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(results[0])
    )
    writer.writeheader()
    writer.writerows(results)

head.forward_single = original_forward
handle.remove()

print("\n=== STEP 11.9.4 COMPLETE ===", flush=True)
print("Report:", REPORT, flush=True)
print("Scenes:", NUM_SCENES, flush=True)
print("No training performed.", flush=True)
