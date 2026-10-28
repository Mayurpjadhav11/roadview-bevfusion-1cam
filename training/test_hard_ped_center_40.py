import random
import types
from pathlib import Path
from datetime import datetime, timezone
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.parallel import DataContainer

from carla_model_loader import build_carla_model
from carla_training_dataset import build_carla_training_dataset
from carla_dataset_config import CLASSES

SEED = 42
STEPS = 40
BASE_LR = 1e-5
OFFSET_LR = 3e-4
PED = list(CLASSES).index("pedestrian")

INITIAL = Path(
    "/outputs/checkpoints/"
    "close_pair_ped_nms_20261028T112941Z/"
    "step10.pth"
)

RUN = datetime.now(timezone.utc).strftime(
    "hard_ped_center_%Y%m%dT%H%M%SZ"
)

OUT = Path("/outputs/experiments") / RUN
CKPT = Path("/outputs/checkpoints") / RUN
OUT.mkdir(parents=True, exist_ok=False)
CKPT.mkdir(parents=True, exist_ok=False)

def unwrap(value):
    return value.data if isinstance(value, DataContainer) else value

def gpu(value, dtype=torch.float32):
    value = unwrap(value)
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.to("cuda", dtype=dtype).contiguous()

MATRICES = [
    "camera2ego", "lidar2ego", "lidar2camera",
    "lidar2image", "camera_intrinsics",
    "camera2lidar", "img_aug_matrix",
    "lidar_aug_matrix",
]

def prepare(sample):
    result = {
        "img": gpu(sample["img"]).unsqueeze(0),
        "points": [gpu(sample["points"])],
        "metas": [unwrap(sample["metas"])],
        "gt_bboxes_3d": [
            unwrap(sample["gt_bboxes_3d"]).to("cuda")
        ],
        "gt_labels_3d": [
            gpu(sample["gt_labels_3d"], torch.long)
        ],
    }
    for key in MATRICES:
        result[key] = gpu(sample[key]).unsqueeze(0)
    return result

def training_mode(model):
    model.train()
    for module in model.modules():
        if isinstance(
            module, torch.nn.modules.batchnorm._BatchNorm
        ):
            module.eval()
        if isinstance(
            module, torch.nn.modules.dropout._DropoutNd
        ):
            module.p = 0.0
        if isinstance(module, torch.nn.MultiheadAttention):
            module.dropout = 0.0

def extract_losses(outputs):
    losses = {}
    for key, value in outputs.items():
        name = key.split("/")[-1]
        if "loss" not in name:
            continue
        if isinstance(value, (list, tuple)):
            value = sum(value)
        losses[name] = value

    required = [
        "loss_heatmap",
        "layer_-1_loss_cls",
        "layer_-1_loss_bbox",
    ]
    for key in required:
        assert key in losses, (key, list(outputs.keys()))
    return losses

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False

saved = torch.load(INITIAL, map_location="cpu")
sample_index = int(saved["sample_index"])

dataset = build_carla_training_dataset()
sample = dataset[sample_index]
data = prepare(sample)

model, _ = build_carla_model(
    pretrained=False,
    device="cpu"
)

head = model.heads["object"]
assert head.num_classes == len(CLASSES)
assert head.num_decoder_layers == 1
assert head.nms_kernel_size == 3

# Reconstruct the architecture added in Step 11.8.44.
head.class_dim_offset = torch.nn.Embedding(
    head.num_classes, 3
)

# Strict loading ensures no experimental parameters
# are silently lost.
model.load_state_dict(saved["model"], strict=True)
model.cuda()
del saved

captured = {}

def dimension_hook(module, inputs, output):
    captured["center"] = output["center"]
    corrections = head.class_dim_offset(
        head.query_labels
    ).transpose(1, 2)

    result = dict(output)
    result["dim"] = output["dim"] + corrections
    return result

dim_hook = head.prediction_heads[0].register_forward_hook(
    dimension_hook
)

# Pedestrian-only heatmap suppression exemption.
original_pool = F.max_pool2d
original_forward = head.forward_single
nms_hits = {"value": 0}

def pedestrian_exempt_pool(x, kernel_size, *args, **kwargs):
    result = original_pool(x, kernel_size, *args, **kwargs)

    if (
        x.ndim == 4
        and x.shape[1] == head.num_classes
        and kernel_size == 3
    ):
        nms_hits["value"] += 1
        result = result.clone()
        result[:, PED, :, :] = x[:, PED, 1:-1, 1:-1]

    return result

def pedestrian_exempt_forward(self, features, img_features, metas):
    with mock.patch.object(
        F, "max_pool2d", new=pedestrian_exempt_pool
    ):
        return original_forward(
            features, img_features, metas
        )

head.forward_single = types.MethodType(
    pedestrian_exempt_forward, head
)

# Preserve the prior experiment's class-consistent
# dimension supervision.
original_targets = head.get_targets
diagnostics = {"mismatches": 0, "positives": 0}

def consistent_targets(*args, **kwargs):
    result = list(original_targets(*args, **kwargs))

    labels = result[0].reshape(-1)
    proposal_classes = head.query_labels.reshape(-1)

    weights = result[3]
    flat = weights.clone().reshape(-1, weights.shape[-1])

    assert len(labels) == len(proposal_classes)

    positive = flat[:, 3:6].sum(1) > 0

    mismatch = (
        positive
        & (labels != proposal_classes)
        & (labels < head.num_classes)
    )

    diagnostics["mismatches"] = int(mismatch.sum().item())
    diagnostics["positives"] = int(positive.sum().item())

    flat[mismatch, 3:6] = 0.0
    result[3] = flat.reshape_as(weights)

    diagnostics["targets"] = result[2].detach()
    diagnostics["weights"] = result[3].detach()
    diagnostics["labels"] = result[0].detach()

    return tuple(result)

head.get_targets = consistent_targets

offset_params = list(head.class_dim_offset.parameters())
offset_ids = {id(p) for p in offset_params}

shared_params = [
    p for p in model.parameters()
    if id(p) not in offset_ids
]

optimizer = torch.optim.AdamW(
    [
        {
            "params": shared_params,
            "lr": BASE_LR,
            "weight_decay": 0.01,
        },
        {
            "params": offset_params,
            "lr": OFFSET_LR,
            "weight_decay": 0.0,
        },
    ]
)

print("=== STEP 11.8.49: CLOSE PAIR FINETUNE ===", flush=True)
print("Source checkpoint:", INITIAL, flush=True)
print("Sample index:", sample_index, flush=True)
print("Pedestrian class:", PED, flush=True)
print("Base LR:", BASE_LR, flush=True)
print("Offset LR:", OFFSET_LR, flush=True)
print("NMS: 3x3 except pedestrian", flush=True)
print("Steps:", STEPS, flush=True)

random.seed(SEED + 149)
np.random.seed(SEED + 149)
torch.manual_seed(SEED + 149)
torch.cuda.manual_seed_all(SEED + 149)

for step in range(1, STEPS + 1):
    training_mode(model)
    optimizer.zero_grad(set_to_none=True)
    nms_hits["value"] = 0

    outputs = model(**data)
    losses = extract_losses(outputs)
    # Extra supervision for the hardest correctly
    # assigned pedestrian, measured in metres.
    pred_center = captured["center"].transpose(
        1, 2
    ).reshape(-1, 2)

    target_tensor = diagnostics["targets"]
    target_center = target_tensor.reshape(
        -1, target_tensor.shape[-1]
    )[:, :2]

    weights = diagnostics["weights"]
    flat_weights = weights.reshape(
        -1, weights.shape[-1]
    )

    labels = diagnostics["labels"].reshape(-1)
    query_classes = head.query_labels.reshape(-1)

    mask = (
        (labels == PED)
        & (query_classes == PED)
        & (flat_weights[:, :2].sum(1) > 0)
    )

    valid_count = int(mask.sum().item())
    assert valid_count >= 2, (
        f"Only {valid_count} class-consistent "
        "pedestrian positives"
    )

    spacing = torch.as_tensor(
        head.bbox_coder.voxel_size[:2],
        device=pred_center.device,
        dtype=pred_center.dtype
    ) * float(head.bbox_coder.out_size_factor)

    center_errors = torch.linalg.vector_norm(
        (pred_center[mask] - target_center[mask])
        * spacing,
        dim=1
    )

    hard_center_error = center_errors.max()
    mean_center_error = center_errors.mean()

    center_aux_loss = (
        1.5 * hard_center_error
        + 0.5 * mean_center_error
    )

    total = sum(losses.values()) + center_aux_loss

    if not torch.isfinite(total).item():
        raise RuntimeError(
            f"Nonfinite loss at step {step}"
        )

    assert nms_hits["value"] > 0, (
        "Pedestrian NMS exemption not applied"
    )

    total.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 35.0
    )

    if not torch.isfinite(grad_norm).item():
        raise RuntimeError(
            f"Nonfinite gradient at step {step}"
        )

    optimizer.step()

    if step == 1 or step % 10 == 0:
        print(
            f"STEP {step:03d}/{STEPS} "
            f"total={total.item():.5f} "
            f"heatmap={losses['loss_heatmap'].item():.5f} "
            f"cls={losses['layer_-1_loss_cls'].item():.5f} "
            f"bbox={losses['layer_-1_loss_bbox'].item():.5f} "
            f"positives={diagnostics['positives']} "
            f"mismatch={diagnostics['mismatches']} "
            f"grad={float(grad_norm):.4f} "
            f"hard_center_m={hard_center_error.item():.4f} "
            f"mean_center_m={mean_center_error.item():.4f} "
            f"center_aux={center_aux_loss.item():.4f}",
            flush=True
        )

    if step % 10 == 0:
        path = CKPT / f"step{step}.pth"
        torch.save(
            {
                "model": {
                    k: v.detach().cpu()
                    for k, v in model.state_dict().items()
                },
                "step": step,
                "sample_index": sample_index,
                "initial_checkpoint": str(INITIAL),
                "class_dim_offset_enabled": True,
                "pedestrian_nms_exempt": True,
                "mismatch_dim_mask_enabled": True,
                "base_lr": BASE_LR,
                "offset_lr": OFFSET_LR,
                "hard_ped_center_loss": True,
                "hard_center_weight": 1.5,
                "mean_center_weight": 0.5,
                "pedestrian_nms_exempt": True,
            },
            path
        )
        print("CHECKPOINT SAVED:", path, flush=True)

print("\nSTEP 11.8.49 COMPLETE", flush=True)
print("Checkpoint directory:", CKPT, flush=True)
print("Experiment directory:", OUT, flush=True)
print("Original BEVFusion source unchanged.", flush=True)

head.forward_single = original_forward
head.get_targets = original_targets
dim_hook.remove()
