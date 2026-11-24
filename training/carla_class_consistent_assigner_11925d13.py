
import torch
from mmdet.core.bbox.assigners import AssignResult


def class_consistent_assign(
    original_assign,
    query_labels,
    bboxes,
    gt_bboxes,
    gt_labels,
    cls_pred,
    train_cfg,
):
    """
    Partition proposals and GTs by class and invoke the
    original Hungarian assigner within each class.

    Unmatched proposals remain background.

    Returns AssignResult with GLOBAL GT indices.
    """

    assert bboxes.ndim == 2
    assert gt_bboxes.ndim == 2
    assert gt_labels.ndim == 1
    assert cls_pred.ndim == 3
    assert cls_pred.shape[0] == 1

    query_labels = query_labels.reshape(-1)

    num_queries = bboxes.shape[0]
    num_gt = gt_bboxes.shape[0]
    num_classes = cls_pred.shape[1]

    assert query_labels.numel() == num_queries
    assert cls_pred.shape[2] == num_queries
    assert gt_labels.numel() == num_gt

    assert bboxes.device == gt_bboxes.device
    assert bboxes.device == gt_labels.device
    assert bboxes.device == cls_pred.device
    assert bboxes.device == query_labels.device

    assert ((query_labels >= 0) &
            (query_labels < num_classes)).all().item()

    assert ((gt_labels >= 0) &
            (gt_labels < num_classes)).all().item()

    assigned_gt_inds = torch.zeros(
        num_queries,
        dtype=torch.long,
        device=bboxes.device,
    )

    assigned_labels = torch.full(
        (num_queries,),
        -1,
        dtype=torch.long,
        device=bboxes.device,
    )

    max_overlaps = bboxes.new_zeros(num_queries)

    for cls in range(num_classes):
        query_idx = torch.nonzero(
            query_labels == cls,
            as_tuple=False,
        ).flatten()

        gt_idx = torch.nonzero(
            gt_labels == cls,
            as_tuple=False,
        ).flatten()

        if query_idx.numel() == 0 or gt_idx.numel() == 0:
            continue

        result = original_assign(
            bboxes.index_select(0, query_idx),
            gt_bboxes.index_select(0, gt_idx),
            gt_labels.index_select(0, gt_idx),
            cls_pred.index_select(2, query_idx),
            train_cfg,
        )

        local_indices = result.gt_inds.long()

        assert local_indices.numel() == query_idx.numel()
        assert result.num_gts == gt_idx.numel()
        assert result.max_overlaps is not None
        assert result.max_overlaps.shape == (query_idx.numel(),)
        assert ((local_indices >= 0) &
                (local_indices <= gt_idx.numel())).all().item()

        positive = local_indices > 0

        if positive.any().item():
            global_query_idx = query_idx[positive]
            local_gt_idx = local_indices[positive] - 1
            global_gt_idx = gt_idx[local_gt_idx]

            assigned_gt_inds[global_query_idx] = (
                global_gt_idx + 1
            )
            assigned_labels[global_query_idx] = (
                gt_labels[global_gt_idx]
            )
            max_overlaps[global_query_idx] = (
                result.max_overlaps[positive]
            )

    positive = assigned_gt_inds > 0

    assert torch.equal(
        assigned_labels[positive],
        query_labels[positive],
    )

    assert torch.all(
        assigned_labels[~positive] == -1
    ).item()

    assert torch.all(
        assigned_gt_inds[~positive] == 0
    ).item()

    if positive.any().item():
        matched_gt = assigned_gt_inds[positive] - 1
        assert torch.unique(matched_gt).numel() == (
            matched_gt.numel()
        )

    assert torch.isfinite(max_overlaps).all().item()

    return AssignResult(
        num_gt,
        assigned_gt_inds,
        max_overlaps,
        labels=assigned_labels,
    )
