
import torch
from scipy.optimize import linear_sum_assignment

from mmdet.core.bbox.assigners import AssignResult

from carla_class_consistent_assigner_11925d13 import (
    class_consistent_assign,
)


# Deterministic synthetic stand-in for the original assigner.
# Tests routing and global-index reconstruction.
# This does not replace the real BEVFusion integration test.
def synthetic_hungarian(
    bboxes,
    gt_bboxes,
    gt_labels,
    cls_pred,
    train_cfg,
):
    assert len(bboxes) > 0
    assert len(gt_bboxes) > 0
    assert cls_pred.shape[2] == len(bboxes)

    cost = torch.cdist(
        bboxes[:, :2],
        gt_bboxes[:, :2],
        p=1,
    ).numpy()

    rows, cols = linear_sum_assignment(cost)

    assigned = torch.zeros(
        len(bboxes), dtype=torch.long
    )
    labels = torch.full(
        (len(bboxes),), -1, dtype=torch.long
    )
    overlaps = torch.zeros(len(bboxes))

    for q, g in zip(rows, cols):
        assigned[q] = int(g) + 1
        labels[q] = gt_labels[g]
        overlaps[q] = 1.0 / (1.0 + float(cost[q, g]))

    return AssignResult(
        len(gt_bboxes),
        assigned,
        overlaps,
        labels=labels,
    )


def run_case(name, query_classes, gt_classes,
             query_xy, gt_xy):
    n = len(query_classes)
    m = len(gt_classes)

    queries = torch.tensor(
        query_classes, dtype=torch.long
    )
    gt_labels = torch.tensor(
        gt_classes, dtype=torch.long
    )

    bboxes = torch.zeros(n, 7)
    gt_bboxes = torch.zeros(m, 7)

    if n:
        bboxes[:, 0] = torch.tensor(
            query_xy, dtype=torch.float32
        )

    if m:
        gt_bboxes[:, 0] = torch.tensor(
            gt_xy, dtype=torch.float32
        )

    scores = torch.zeros(1, 6, n)

    result = class_consistent_assign(
        synthetic_hungarian,
        queries,
        bboxes,
        gt_bboxes,
        gt_labels,
        scores,
        None,
    )

    assert result.num_gts == m
    assert result.gt_inds.numel() == n
    assert result.max_overlaps.numel() == n

    positive = result.gt_inds > 0
    matched_count = int(positive.sum().item())

    assert matched_count == sum(
        min(
            query_classes.count(cls),
            gt_classes.count(cls),
        )
        for cls in range(6)
    )

    if matched_count:
        matched = gt_labels[
            result.gt_inds[positive] - 1
        ]
        assert torch.equal(
            matched, queries[positive]
        )

    print(
        "PASS:",
        name,
        "queries=", n,
        "GT=", m,
        "matches=", matched_count,
        flush=True,
    )

    return result


# No GT, ordinary 200-proposal frame.
run_case(
    "EMPTY GT",
    [0] * 100 + [4] * 100,
    [],
    list(range(200)),
    [],
)

# Query class and GT class do not overlap.
run_case(
    "NO ELIGIBLE CLASS",
    [0, 0, 0],
    [4, 4],
    [0, 1, 2],
    [0, 1],
)

# Cross-class spatially closer candidates.
# GT assignment must still respect query class.
r = run_case(
    "CROSS-CLASS NEAREST CONFLICT",
    [4, 0],
    [0, 4],
    [0, 100],
    [0, 100],
)
assert r.gt_inds.tolist() == [2, 1]
assert r.labels.tolist() == [4, 0]

# Two same-class GT objects, only one eligible query.
run_case(
    "GT EXCEEDS QUERY CAPACITY",
    [4],
    [4, 4],
    [0],
    [0, 1],
)

# Multiple queries competing for one GT.
run_case(
    "QUERY COMPETITION",
    [4, 4, 4],
    [4],
    [0, 1, 2],
    [1.1],
)

# Mixed-class case with some classes absent.
run_case(
    "MIXED SIX-CLASS LABEL SPACE",
    [0, 4, 4, 2, 5, 3, 0],
    [4, 0, 2, 3],
    [0, 1, 2, 3, 4, 5, 6],
    [1, 0, 3, 5],
)

# Standard proposal capacity, multiple GT.
run_case(
    "200-PROPOSAL CONTRACT",
    [0] * 80 + [4] * 70 + [2] * 50,
    [0, 4, 4, 2, 0],
    list(range(200)),
    [10, 90, 100, 160, 20],
)

print()
print("SYNTHETIC MATCHER CONTRACT: PASS")
print("UPSTREAM MATCHER: UNCHANGED")
print("TRAINING EXECUTED: NONE")
