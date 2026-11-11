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

import os

PED_NMS_MODE = os.environ["CARLA_PED_NMS_EXEMPT"]
assert PED_NMS_MODE in ("0", "1")
PED_EXEMPT = PED_NMS_MODE == "1"

SIZE_PRIOR = os.environ.get("CARLA_SIZE_PRIOR", "1")
assert SIZE_PRIOR in ("0", "1")
USE_SIZE_PRIOR = SIZE_PRIOR == "1"
assert not USE_SIZE_PRIOR, (
    "Score audit requires original model predictions"
)


# Class medians from the same 12 training frames.
# Diagnostic oracle only; not learned model parameters.
CLASS_SIZE_PRIORS = [
    [1.930, 4.792, 1.549],   # car
    [2.627, 5.204, 2.467],   # truck
    [3.944, 10.273, 4.253], # bus
    [0.797, 2.044, 1.590],   # motorcycle
    [0.866, 1.509, 1.612],   # bicycle
    [0.375, 0.375, 1.860],   # pedestrian
]

SEED = 119
NUM_SCENES = 24
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
    (
        "XY_RESTRICTED30",
        ROOT /
        "carla_xyrestricted_pilot_20261029T093851Z_smoke/"
        "step30.pth",
    ),
]

REPORT = Path(
    "/outputs/experiments/"
    "carla_near_gt_scores_val24_119221f.csv"
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

print("=== STEP 11.9.6: DETECTION VALIDATION ===", flush=True)

dataset = build_dataset("val")
assert len(dataset) == 1800
assert dataset.filter_empty_gt is False


# Select frames using validation annotations only.
# The class-size priors remain fixed from training.
records = []

for index in range(len(dataset)):
    token = str(dataset.data_infos[index]["token"])
    scene = token.rsplit("_", 1)[0]

    ann = None
    if hasattr(dataset, "get_ann_info"):
        ann = dataset.get_ann_info(index)

    if isinstance(ann, dict) and "gt_labels_3d" in ann:
        labels = torch.as_tensor(
            ann["gt_labels_3d"]
        ).long().tolist()
    else:
        sample = dataset[index]
        if sample is None:
            continue
        labels = torch.as_tensor(
            unwrap(sample["gt_labels_3d"])
        ).long().tolist()

    labels = {
        int(label) for label in labels
        if 0 <= int(label) < len(CLASSES)
    }

    records.append({
        "index": index,
        "token": token,
        "scene": scene,
        "labels": labels,
    })

available = set().union(
    *(r["labels"] for r in records)
)

assert available == set(range(len(CLASSES))), (
    "Validation split missing classes:",
    [CLASSES[c] for c in
     sorted(set(range(len(CLASSES))) - available)]
)

# Rare classes first.
class_order = sorted(
    range(len(CLASSES)),
    key=lambda cls: sum(
        cls in r["labels"] for r in records
    )
)

selected = []
used_indices = set()
used_scenes = set()
covered = set()

for cls in class_order:
    if cls in covered:
        continue

    choices = [
        r for r in records
        if cls in r["labels"]
        and r["index"] not in used_indices
    ]

    assert choices, CLASSES[cls]

    # Prefer a new scene and multiple uncovered classes.
    choices.sort(
        key=lambda r: (
            r["scene"] not in used_scenes,
            len(r["labels"] - covered),
        ),
        reverse=True,
    )

    chosen = choices[0]
    selected.append(chosen)
    used_indices.add(chosen["index"])
    used_scenes.add(chosen["scene"])
    covered.update(chosen["labels"])

assert covered == set(range(len(CLASSES)))

# Fill remaining positions with different scenes first.
remaining = [
    r for r in records
    if r["index"] not in used_indices
]
random.Random(SEED).shuffle(remaining)

for record in remaining:
    if len(selected) >= NUM_SCENES:
        break
    if record["scene"] in used_scenes:
        continue

    selected.append(record)
    used_indices.add(record["index"])
    used_scenes.add(record["scene"])

# Permit repeated scenes only if required to reach 24.
for record in remaining:
    if len(selected) >= NUM_SCENES:
        break
    if record["index"] in used_indices:
        continue

    selected.append(record)
    used_indices.add(record["index"])

indices = [r["index"] for r in selected]
prefixes = {r["scene"] for r in selected}

assert len(indices) == NUM_SCENES
assert len(set(indices)) == NUM_SCENES

print("VAL24 tokens:",
      [r["token"] for r in selected], flush=True)
print("VAL24 unique scenes:", len(prefixes), flush=True)
print("VAL24 class coverage:",
      [CLASSES[c] for c in sorted(covered)], flush=True)

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
    if not PED_EXEMPT:
        if (
            x.ndim == 4
            and x.shape[1] == head.num_classes
            and kernel_size == 3
        ):
            nms_calls["count"] += 1
        return original_pool(
            x, kernel_size, *args, **kwargs
        )
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


# ============================================================
# SCORE COMPONENT AUDIT
# ============================================================

score_samples = []

original_get_bboxes = head.get_bboxes

def score_audit_get_bboxes(
    self, preds_dicts, metas, *args, **kwargs
):

    raw = preds_dicts[0][0]

    query = (
        self.query_labels.detach()
        .long()
    )

    decoder = (
        raw["heatmap"][
            ..., -self.num_proposals:
        ].detach().sigmoid()
    )

    bev = (
        raw["query_heatmap_score"]
        .detach()
    )

    assert decoder.shape == bev.shape
    assert decoder.shape[0] == 1
    assert decoder.shape[1] == len(CLASSES)
    assert decoder.shape[2] == self.num_proposals

    assert query.shape == (
        1, self.num_proposals
    )

    # Select score of the BEV-assigned query class.
    indices = query.unsqueeze(1)

    selected_decoder = (
        decoder.gather(1, indices)
        .squeeze(1)
    )

    selected_bev = (
        bev.gather(1, indices)
        .squeeze(1)
    )

    selected_final = (
        selected_decoder * selected_bev
    )

    decoder_argmax = decoder.argmax(dim=1)

    assert torch.isfinite(selected_decoder).all()
    assert torch.isfinite(selected_bev).all()
    assert torch.isfinite(selected_final).all()

    assert (
        (selected_decoder >= 0)
        & (selected_decoder <= 1)
    ).all()

    assert (
        (selected_bev >= 0)
        & (selected_bev <= 1)
    ).all()

    score_samples.append({
        "query": query[0].cpu().numpy().copy(),
        "decoder": (
            selected_decoder[0]
            .cpu().numpy().copy()
        ),
        "bev": (
            selected_bev[0]
            .cpu().numpy().copy()
        ),
        "final": (
            selected_final[0]
            .cpu().numpy().copy()
        ),
        "argmax": (
            decoder_argmax[0]
            .cpu().numpy().copy()
        ),
    })

    # Execute original inference logic unchanged.
    return original_get_bboxes(
        preds_dicts, metas, *args, **kwargs
    )

head.get_bboxes = types.MethodType(
    score_audit_get_bboxes,
    head
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

    query_total = np.zeros(len(CLASSES), dtype=np.int64)
    pred_total = np.zeros(len(CLASSES), dtype=np.int64)
    gt_total = np.zeros(len(CLASSES), dtype=np.int64)
    missing_query_gt = np.zeros(len(CLASSES), dtype=np.int64)
    confident_total = np.zeros(len(CLASSES), dtype=np.int64)

    predicted_dims = [[] for _ in CLASSES]
    target_dims = [[] for _ in CLASSES]
    matching_query_labels = 0
    compared_query_labels = 0

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

        if USE_SIZE_PRIOR:
            box_obj = prediction["boxes_3d"]
            labels = prediction["labels_3d"].long()

            priors = torch.tensor(
                CLASS_SIZE_PRIORS,
                dtype=box_obj.tensor.dtype,
                device=box_obj.tensor.device,
            )

            assert priors.shape == (len(CLASSES), 3)
            assert box_obj.tensor.shape[1] >= 7
            assert len(labels) == len(box_obj.tensor)
            assert torch.all(
                (labels >= 0) & (labels < len(CLASSES))
            ).item()

            modified = box_obj.tensor.clone()
            modified[:, 3:6] = priors[labels]

            # Modify only this decoded prediction object.
            # Model parameters and checkpoint are unchanged.
            box_obj.tensor = modified

            assert torch.allclose(
                box_obj.tensor[:, 3:6],
                priors[labels],
            )

            if iteration == 1:
                print(
                    "SIZE_ORACLE_ACTIVE",
                    model_name,
                    "boxes=", len(labels),
                    flush=True,
                )

        assert nms_calls["count"] > 0

        # -----------------------------------------------
        # Near-GT proposal classification.
        # This does not modify inference or box scores.
        # -----------------------------------------------
        assert len(score_samples) == iteration

        sample_scores = score_samples[-1]
        query = sample_scores["query"]

        predicted_boxes = prediction["boxes_3d"]
        predicted_labels = (
            prediction["labels_3d"]
            .detach().cpu().numpy().astype(int)
        )

        predicted_xy = (
            predicted_boxes.corners
            .detach().cpu().numpy()
            .mean(axis=1)[:, :2]
        )

        gt_xy = gt_corners.mean(axis=1)[:, :2]

        assert len(query) == head.num_proposals
        assert len(predicted_xy) == len(query)
        assert len(predicted_labels) == len(query)

        near = np.zeros(len(query), dtype=bool)

        for g, cls in enumerate(gt_labels):
            cls = int(cls)

            candidates = np.flatnonzero(
                (query == cls)
                & (predicted_labels == cls)
            )

            if not len(candidates):
                continue

            distances = np.linalg.norm(
                predicted_xy[candidates] - gt_xy[g],
                axis=1
            )

            near[candidates[distances <= 2.0]] = True

        sample_scores["near_gt"] = near
        sample_scores["label_mismatches"] = int(
            (query != predicted_labels).sum()
        )


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

        queries = (
            head.query_labels.detach()
            .reshape(-1).cpu().numpy().astype(int)
        )

        assert len(queries) == head.num_proposals
        assert np.all((queries >= 0) & (queries < len(CLASSES)))

        assert np.all(
            (pred_labels >= 0) &
            (pred_labels < len(CLASSES))
        )

        q_hist = np.bincount(
            queries, minlength=len(CLASSES)
        )
        p_hist = np.bincount(
            pred_labels, minlength=len(CLASSES)
        )
        g_hist = np.bincount(
            gt_labels, minlength=len(CLASSES)
        )

        query_total += q_hist
        pred_total += p_hist
        gt_total += g_hist

        missing_query_gt += np.bincount(
            gt_labels[q_hist[gt_labels] == 0],
            minlength=len(CLASSES)
        )

        confident_total += np.bincount(
            pred_labels[pred_scores >= 0.005],
            minlength=len(CLASSES)
        )

        if len(queries) == len(pred_labels):
            matching_query_labels += int(
                (queries == pred_labels).sum()
            )
            compared_query_labels += len(queries)

        pdims = (
            prediction["boxes_3d"].tensor
            .detach().cpu().numpy()[:, 3:6]
        )
        gdims = (
            gt_boxes.tensor.detach()
            .cpu().numpy()[:, 3:6]
        )

        assert len(pdims) == len(pred_labels)
        assert len(gdims) == len(gt_labels)

        for cls in range(len(CLASSES)):
            predicted_dims[cls].extend(
                pdims[pred_labels == cls]
            )
            target_dims[cls].extend(
                gdims[gt_labels == cls]
            )

        overlaps = overlap_matrix(
            gt_corners, gt_labels,
            pred_corners, pred_labels
        )


        # STEP 11.9.21.5 — diagnostic only.
        # No changes to the original detection metric.
        token = str(dataset.data_infos[index]["token"])

        gt_centers = gt_corners.mean(axis=1)
        pred_centers = pred_corners.mean(axis=1)

        gt_parameters = gt_boxes.tensor.detach().cpu().numpy()
        pred_parameters = (
            prediction["boxes_3d"].tensor
            .detach().cpu().numpy()
        )

        for g in range(len(gt_labels)):
            gt_class = int(gt_labels[g])

            # Remove only the class restriction from
            # the original rotated-3D-IoU evaluator.
            fake_labels = np.full_like(
                pred_labels, gt_class
            )

            agnostic_ious = overlap_matrix(
                gt_corners[g:g+1],
                gt_labels[g:g+1],
                pred_corners,
                fake_labels,
            )[0]

            best_idx = int(np.argmax(agnostic_ious))
            best_iou = float(agnostic_ious[best_idx])

            distances = np.linalg.norm(
                pred_centers - gt_centers[g],
                axis=1,
            )

            nearest_idx = int(np.argmin(distances))
            nearest_distance = float(
                distances[nearest_idx]
            )

            same_class = np.flatnonzero(
                pred_labels == gt_class
            )

            if len(same_class):
                best_correct_iou = float(
                    agnostic_ious[same_class].max()
                )
            else:
                best_correct_iou = 0.0

            best_class = CLASSES[int(pred_labels[best_idx])]
            nearest_class = CLASSES[
                int(pred_labels[nearest_idx])
            ]

            gt_xyz = gt_centers[g]
            nearest_xyz = pred_centers[nearest_idx]

            gt_dims = gt_parameters[g, 3:6]
            nearest_dims = pred_parameters[
                nearest_idx, 3:6
            ]

            print(
                f"GEO model={model_name} "
                f"token={token} "
                f"gt_index={g} "
                f"true_class={CLASSES[gt_class]} "
                f"best_any_iou={best_iou:.5f} "
                f"best_any_class={best_class} "
                f"best_any_score={float(pred_scores[best_idx]):.6f} "
                f"best_correct_iou={best_correct_iou:.5f} "
                f"nearest_m={nearest_distance:.3f} "
                f"nearest_class={nearest_class} "
                f"nearest_score={float(pred_scores[nearest_idx]):.6f} "
                f"gt_xyz={np.round(gt_xyz,2).tolist()} "
                f"nearest_xyz={np.round(nearest_xyz,2).tolist()} "
                f"gt_dims={np.round(gt_dims,2).tolist()} "
                f"nearest_dims={np.round(nearest_dims,2).tolist()}",
                flush=True,
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

    print(
        f"QSUMMARY model={model_name} "
        f"GT={int(gt_total.sum())} "
        f"queries={int(query_total.sum())} "
        f"predictions={int(pred_total.sum())} "
        f"GT_without_matching_query={int(missing_query_gt.sum())} "
        f"query_pred_label_agreement="
        f"{matching_query_labels}/{compared_query_labels}",
        flush=True
    )

    for cls, name in enumerate(CLASSES):
        if predicted_dims[cls]:
            pred_median = np.median(
                np.asarray(predicted_dims[cls]), axis=0
            ).round(3).tolist()
        else:
            pred_median = None

        if target_dims[cls]:
            gt_median = np.median(
                np.asarray(target_dims[cls]), axis=0
            ).round(3).tolist()
        else:
            gt_median = None

        print(
            f"QCLASS model={model_name} "
            f"class={name} "
            f"GT={gt_total[cls]} "
            f"queries={query_total[cls]} "
            f"predictions={pred_total[cls]} "
            f"score005={confident_total[cls]} "
            f"GT_no_query={missing_query_gt[cls]} "
            f"GT_dims={gt_median} "
            f"pred_dims={pred_median}",
            flush=True
        )

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

print("\n=== STEP 11.9.6 COMPLETE ===", flush=True)
print("Report:", REPORT, flush=True)
print("Scenes:", NUM_SCENES, flush=True)
print("No training performed.", flush=True)


# ============================================================
# SCORE COMPONENT REPORT
# ============================================================

print(
    "\n=== STEP 11.9.22.1C SCORE COMPONENT REPORT ===",
    flush=True
)

assert len(score_samples) == NUM_SCENES, (
    "Expected one captured query batch per frame",
    len(score_samples),
    NUM_SCENES
)

queries = np.concatenate([
    sample["query"] for sample in score_samples
])

decoders = np.concatenate([
    sample["decoder"] for sample in score_samples
])

bevs = np.concatenate([
    sample["bev"] for sample in score_samples
])

finals = np.concatenate([
    sample["final"] for sample in score_samples
])

argmaxes = np.concatenate([
    sample["argmax"] for sample in score_samples
])

assert len(queries) == NUM_SCENES * head.num_proposals
assert np.isfinite(decoders).all()
assert np.isfinite(bevs).all()
assert np.isfinite(finals).all()

assert np.allclose(
    finals,
    decoders * bevs,
    rtol=1e-6,
    atol=1e-7
)

for cls, name in enumerate(CLASSES):

    mask = queries == cls
    count = int(mask.sum())

    if count == 0:
        print(
            "SCORE_CLASS class={} queries=0".format(name),
            flush=True
        )
        continue

    dec = decoders[mask]
    bev = bevs[mask]
    final = finals[mask]

    agreement = float(
        np.mean(argmaxes[mask] == cls)
    )

    above005 = int((final >= 0.005).sum())
    above010 = int((final >= 0.010).sum())

    print(
        "SCORE_CLASS "
        "class={} "
        "queries={} "
        "decoder_median={:.6f} "
        "decoder_p90={:.6f} "
        "bev_median={:.6f} "
        "bev_p90={:.6f} "
        "final_median={:.6f} "
        "final_p90={:.6f} "
        "final_ge005={} "
        "final_ge010={} "
        "argmax_agreement={:.4f}".format(
            name,
            count,
            float(np.median(dec)),
            float(np.percentile(dec, 90)),
            float(np.median(bev)),
            float(np.percentile(bev, 90)),
            float(np.median(final)),
            float(np.percentile(final, 90)),
            above005,
            above010,
            agreement
        ),
        flush=True
    )

print(
    "SCORE_TOTAL queries={} "
    "frames={} "
    "final_ge005={} "
    "final_ge010={}".format(
        len(queries),
        len(score_samples),
        int((finals >= 0.005).sum()),
        int((finals >= 0.010).sum())
    ),
    flush=True
)

print("SCORE_COMPONENT_AUDIT: PASS", flush=True)


# ============================================================
# STEP 11.9.22.1F - NEAR-GT CONFIDENCE REPORT
# ============================================================

print(
    "\n=== NEAR-GT VS OTHER PROPOSAL SCORES ===",
    flush=True
)

near_all = np.concatenate([
    item["near_gt"] for item in score_samples
])

query_all = np.concatenate([
    item["query"] for item in score_samples
])

decoder_all = np.concatenate([
    item["decoder"] for item in score_samples
])

bev_all = np.concatenate([
    item["bev"] for item in score_samples
])

final_all = np.concatenate([
    item["final"] for item in score_samples
])

argmax_all = np.concatenate([
    item["argmax"] for item in score_samples
])

mismatches = sum(
    item["label_mismatches"]
    for item in score_samples
)

assert len(near_all) == NUM_SCENES * head.num_proposals
assert np.isfinite(final_all).all()

for cls, name in enumerate(CLASSES):
    for group, mask in (
        ("near_gt", near_all),
        ("other", ~near_all),
    ):
        selected = (query_all == cls) & mask
        count = int(selected.sum())

        if not count:
            print(
                "NEAR_SCORE class={} group={} n=0".format(
                    name, group
                ),
                flush=True
            )
            continue

        d = decoder_all[selected]
        b = bev_all[selected]
        f = final_all[selected]
        a = argmax_all[selected]

        print(
            "NEAR_SCORE "
            "class={} group={} n={} "
            "decoder_med={:.6f} "
            "bev_med={:.6f} "
            "final_med={:.6f} "
            "final_p90={:.6f} "
            "ge005={} ge010={} "
            "decoder_ped_fraction={:.4f}".format(
                name,
                group,
                count,
                float(np.median(d)),
                float(np.median(b)),
                float(np.median(f)),
                float(np.percentile(f, 90)),
                int((f >= 0.005).sum()),
                int((f >= 0.010).sum()),
                float(np.mean(
                    a == list(CLASSES).index("pedestrian")
                )),
            ),
            flush=True
        )

print(
    "NEAR_TOTAL queries={} near_gt={} other={} "
    "decoded_label_mismatches={}".format(
        len(near_all),
        int(near_all.sum()),
        int((~near_all).sum()),
        mismatches
    ),
    flush=True
)

print("NEAR_SCORE_AUDIT: PASS", flush=True)
