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
    "Step 47 must evaluate actual model dimensions"
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
NUM_SCENES = 120
IOU_THRESHOLD = 0.25
SCORE_THRESHOLDS = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1]

PED = list(CLASSES).index("pedestrian")
CAR = list(CLASSES).index("car")

ROOT = Path("/outputs/checkpoints")
SOURCE = ROOT / (
    "hard_ped_center_fixed_20261028T114303Z/step40.pth"
)
PILOT = ROOT / "carla_multiscene_20_20261028T133530Z"

CHECKPOINTS = [
    ('COMMON1500', ROOT / 'carla_full_epoch1_11925d/step1500.pth'),
    ('A_ORIGINAL2000', ROOT / 'carla_ab_original_11925d15/step2000.pth'),
    ('B_CLASSCONSISTENT2000', ROOT / 'carla_ab_classconsistent_11925d15/step2000.pth'),
    ('B_CLASSCONSISTENT2500', ROOT / 'carla_branch_b_continuation_11925d41/step2500.pth'),
]

REPORT = Path('/outputs/experiments/carla_b2500_comparison_11925d44_detection.csv')

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


# STEP 11.9.22.5M: read-only initial query capture.

query_capture = {
    "calls": 0,
    "positions": None,
}

query_rows = []
first_model_queries = {}
query_change_rows = []

QUERY_REPORT = Path('/outputs/experiments/carla_b2500_comparison_11925d44_query.csv')

assert not QUERY_REPORT.exists(), QUERY_REPORT
assert not REPORT.exists(), REPORT

def capture_initial_queries(module, args):
    assert len(args) >= 3

    pos = args[2]

    assert pos.ndim == 3, pos.shape
    assert pos.shape[0] == 1, pos.shape
    assert pos.shape[1] == head.num_proposals
    assert pos.shape[2] == 2

    query_capture["positions"] = (
        pos.detach().cpu().numpy().copy()
    )

    query_capture["calls"] += 1

query_hook = head.decoder[0].register_forward_pre_hook(
    capture_initial_queries
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
        query_capture["calls"] = 0
        query_capture["positions"] = None

        with torch.no_grad():
            prediction = model(**inputs)[0]

        assert query_capture["calls"] == 1, (
            "Expected exactly one decoder call",
            query_capture["calls"],
        )

        initial_cells = query_capture["positions"][0]

        initial_labels = (
            head.query_labels.detach()
            .cpu().numpy().reshape(-1).astype(int)
        )

        assert initial_cells.shape == (
            head.num_proposals, 2
        )
        assert initial_labels.shape == (
            head.num_proposals,
        )

        assert np.isfinite(initial_cells).all()

        spacing = np.asarray(
            head.bbox_coder.voxel_size[:2],
            dtype=np.float64
        ) * float(head.bbox_coder.out_size_factor)

        origin = np.asarray(
            head.bbox_coder.pc_range[:2],
            dtype=np.float64
        )

        assert np.all(spacing > 0)

        # Convert initial BEV-cell coordinates into
        # LiDAR-frame XY metres.
        initial_xy = (
            initial_cells.astype(np.float64)
            * spacing[None, :]
            + origin[None, :]
        )

        token_query = str(
            dataset.data_infos[index]["token"]
        )

        if model_name == "COMMON1500":
            assert token_query not in first_model_queries

            first_model_queries[token_query] = (
                initial_xy.copy(),
                initial_labels.copy(),
            )

        elif model_name == "B_CLASSCONSISTENT2000":
            assert token_query in first_model_queries

            base_xy, base_labels = first_model_queries[
                token_query
            ]

            # Compare class+position proposal sets.
            # Proposal ordering can change after heatmap training.
            def proposal_set(xy, labels):
                return {
                    (
                        int(label),
                        round(float(point[0]), 4),
                        round(float(point[1]), 4),
                    )
                    for point, label in zip(xy, labels)
                }

            base_set = proposal_set(base_xy, base_labels)
            new_set = proposal_set(initial_xy, initial_labels)

            assert len(base_set) == head.num_proposals
            assert len(new_set) == head.num_proposals

            common = len(base_set & new_set)
            changed = head.num_proposals - common

            base_ped = int((base_labels == PED).sum())
            new_ped = int((initial_labels == PED).sum())

            query_change_rows.append({
                "token": token_query,
                "common": common,
                "changed": changed,
                "base_ped": base_ped,
                "new_ped": new_ped,
            })

            print(
                "QCHANGE token={} common={} changed={} "
                "base_ped_queries={} new_ped_queries={}".format(
                    token_query,
                    common,
                    changed,
                    base_ped,
                    new_ped,
                ),
                flush=True,
            )

        final_xy = (
            prediction["boxes_3d"].tensor[:, :2]
            .detach().cpu().numpy()
            .astype(np.float64)
        )

        final_labels = (
            prediction["labels_3d"]
            .detach().cpu().numpy()
            .reshape(-1).astype(int)
        )

        assert final_xy.shape == initial_xy.shape
        assert len(final_labels) == len(initial_labels)
        assert np.array_equal(
            final_labels, initial_labels
        ), "Decoded query labels differ from initial labels"

        gt_xy = (
            gt_boxes.tensor[:, :2]
            .detach().cpu().numpy()
            .astype(np.float64)
        )

        for g, cls in enumerate(gt_labels):
            cls = int(cls)

            init_d = np.linalg.norm(
                initial_xy - gt_xy[g][None, :],
                axis=1,
            )

            final_d = np.linalg.norm(
                final_xy - gt_xy[g][None, :],
                axis=1,
            )

            candidates = np.flatnonzero(
                initial_labels == cls
            )

            init_any = float(init_d.min())

            init_same = (
                float(init_d[candidates].min())
                if len(candidates)
                else float("nan")
            )

            final_same = (
                float(final_d[candidates].min())
                if len(candidates)
                else float("nan")
            )

            n05 = int(
                (init_d[candidates] <= 0.5).sum()
            )

            n10 = int(
                (init_d[candidates] <= 1.0).sum()
            )

            n20 = int(
                (init_d[candidates] <= 2.0).sum()
            )

            row = {
                "model": model_name,
                "token": token_query,
                "gt_index": g,
                "class": CLASSES[cls],
                "initial_same_xy_m": init_same,
                "initial_any_xy_m": init_any,
                "final_same_xy_m": final_same,
                "initial_same_within_05": n05,
                "initial_same_within_10": n10,
                "initial_same_within_20": n20,
                "same_class_query_count": len(candidates),
            }

            query_rows.append(row)

            print(
                "QINIT model={} token={} gt={} "
                "class={} initial_same_xy={:.5f} "
                "initial_any_xy={:.5f} "
                "final_same_xy={:.5f} "
                "near05={} near10={} near20={} "
                "same_queries={}".format(
                    model_name,
                    token_query,
                    g,
                    CLASSES[cls],
                    init_same,
                    init_any,
                    final_same,
                    n05,
                    n10,
                    n20,
                    len(candidates),
                ),
                flush=True,
            )

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


# STEP 11.9.22.5M — Query coverage summary.


# Determine GT counts from the completed detection results.
# These counts must agree with GT-level query coverage.
expected_gt_by_model = {}

for checkpoint_name, _ in CHECKPOINTS:
    gt_count = sum(
        int(row["tp"]) + int(row["fn"])
        for row in results
        if row["model"] == checkpoint_name
        and float(row["score_threshold"]) == 0.0
    )

    assert gt_count >= 0
    expected_gt_by_model[checkpoint_name] = gt_count

assert len(query_rows) == sum(
    expected_gt_by_model.values()
), (
    "Query coverage count mismatch",
    len(query_rows),
    expected_gt_by_model,
)

assert len(first_model_queries) == NUM_SCENES

print("\n=== INITIAL QUERY COVERAGE SUMMARY ===")

for name, _ in CHECKPOINTS:
    model_rows = [
        r for r in query_rows
        if r["model"] == name
    ]

    assert len(model_rows) == expected_gt_by_model[name], (name, len(model_rows), expected_gt_by_model[name])

    for cls in CLASSES:
        rows = [
            r for r in model_rows
            if r["class"] == cls
        ]

        initial = np.asarray([
            r["initial_same_xy_m"]
            for r in rows
        ], dtype=float)

        final = np.asarray([
            r["final_same_xy_m"]
            for r in rows
        ], dtype=float)

        valid_initial = initial[np.isfinite(initial)]
        valid_final = final[np.isfinite(final)]

        med_initial = (
            float(np.median(valid_initial))
            if len(valid_initial)
            else float("nan")
        )

        med_final = (
            float(np.median(valid_final))
            if len(valid_final)
            else float("nan")
        )

        print(
            "QSUMMARY model={} class={} GT={} "
            "initial_median={:.5f} "
            "final_median={:.5f} "
            "initial_near05={} "
            "initial_near10={} "
            "initial_near20={}".format(
                name,
                cls,
                len(rows),
                med_initial,
                med_final,
                sum(
                    r["initial_same_within_05"] > 0
                    for r in rows
                ),
                sum(
                    r["initial_same_within_10"] > 0
                    for r in rows
                ),
                sum(
                    r["initial_same_within_20"] > 0
                    for r in rows
                ),
            ),
            flush=True,
        )

assert not QUERY_REPORT.exists()

with QUERY_REPORT.open("x", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(query_rows[0])
    )
    writer.writeheader()
    writer.writerows(query_rows)

assert len(query_change_rows) == NUM_SCENES
assert len({
    r["token"] for r in query_change_rows
}) == NUM_SCENES

print("\n=== HEATMAP QUERY CHANGE SUMMARY ===")

print(
    "FRAMES_WITH_CHANGED_PROPOSALS:",
    sum(r["changed"] > 0 for r in query_change_rows)
)

print(
    "TOTAL_REPLACED_PROPOSALS:",
    sum(r["changed"] for r in query_change_rows)
)

print(
    "BASE_PEDESTRIAN_QUERY_TOTAL:",
    sum(r["base_ped"] for r in query_change_rows)
)

print(
    "NEW_PEDESTRIAN_QUERY_TOTAL:",
    sum(r["new_ped"] for r in query_change_rows)
)

print("QUERY_CHANGE_AUDIT: PASS", flush=True)

query_hook.remove()

print("\nINITIAL_QUERY_COVERAGE: PASS", flush=True)
print("QUERY_REPORT:", QUERY_REPORT, flush=True)
print("QUERY_GT_RECORDS:", len(query_rows), flush=True)
print("HEATMAP_QUERY_COMPARISON: PASS", flush=True)
