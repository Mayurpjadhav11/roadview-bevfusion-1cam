import os
import random
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
from mmcv.parallel import DataContainer

from carla_dataset_config import build_dataset, CLASSES
from carla_model_loader import build_carla_model

SEED = 42
STEPS = 16200
LR = 1e-5

SOURCE = Path(
    "/outputs/checkpoints/"
    "carla_dimcal_train120_zero_update_1192146/"
    "calibrated_step30_no_updates.pth"
)

assert SOURCE.is_file()


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", required=True)
parser.add_argument("--limit", type=int, default=16200)
parser.add_argument("--stop-after", type=int, required=True)
parser.add_argument("--save-every", type=int, default=500)
parser.add_argument("--resume", default=None)

args = parser.parse_args()

STEPS = args.limit

assert 1 <= STEPS <= 16200
assert 1 <= args.stop_after <= STEPS
assert args.save_every >= 1

OUT = Path(args.run_dir).resolve()

assert OUT.parent == Path(
    "/outputs/checkpoints"
).resolve(), OUT

if args.resume is None:
    assert not OUT.exists(), (
        "Run directory already exists",
        OUT,
    )
    OUT.mkdir(parents=True, exist_ok=False)
else:
    assert OUT.is_dir(), OUT
    assert Path(args.resume).is_file()

RUN = OUT.name

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)

print("DETERMINISTIC ALGORITHMS: ENABLED", flush=True)


def unwrap(x):
    return x.data if isinstance(x, DataContainer) else x

def gpu(x, dtype=torch.float32):
    x = unwrap(x)
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return x.to("cuda", dtype=dtype).contiguous()

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

    for key in (
        "camera2ego", "lidar2ego", "lidar2camera",
        "lidar2image", "camera_intrinsics",
        "camera2lidar", "img_aug_matrix",
        "lidar_aug_matrix",
    ):
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

print("=== STEP 11.9.24.3 ===", flush=True)
print("FULL-MODEL TRAINING SMOKE TEST", flush=True)
print("Run:", RUN, flush=True)

dataset = build_dataset("train")
assert len(dataset) == 16200




# Deterministic frame order, independently generated
# from the global RNG state.
assert len(dataset) == 16200

indices = list(range(len(dataset)))
schedule_rng = random.Random(SEED + 4219)
schedule_rng.shuffle(indices)

indices = indices[:STEPS]

assert len(indices) == STEPS
assert len(set(indices)) == STEPS

selected_scenes = [
    str(dataset.data_infos[i]["token"]).rsplit("_", 1)[0]
    for i in indices
]

schedule_hash = hashlib.sha256(
    ",".join(str(i) for i in indices).encode()
).hexdigest()

print("RUN:", RUN, flush=True)
print("TRAINING FRAMES:", STEPS, flush=True)
print("SCHEDULE SHA256:", schedule_hash, flush=True)
print("UNIQUE SCENES:", len(set(selected_scenes)), flush=True)

model, _ = build_carla_model(
    pretrained=False, device="cpu"
)

head = model.heads["object"]

assert head.num_classes == len(CLASSES)
assert head.nms_kernel_size == 3

head.class_dim_offset = torch.nn.Embedding(
    head.num_classes, 3
)

state = torch.load(SOURCE, map_location="cpu")
model.load_state_dict(state["model"], strict=True)

assert state["dimension_calibration"][
    "additional_optimizer_updates"
] == 0

del state

def dimension_hook(module, inputs, output):
    result = dict(output)
    correction = head.class_dim_offset(
        head.query_labels
    ).transpose(1, 2)

    result["dim"] = output["dim"] + correction
    return result

dim_handle = head.prediction_heads[0].register_forward_hook(
    dimension_hook
)

# Preserve the previous class-consistent dimension
# supervision, without the pedestrian NMS exemption.

# Repair the empty-GT assignment contract without changing
# matching, labels or overlaps for nonempty-GT frames.
assert head.train_cfg.assigner.type == "HungarianAssigner3D"

original_assign = head.bbox_assigner.assign
empty_assignments = {"count": 0}

def empty_gt_safe_assign(*args, **kwargs):
    assert len(args) >= 2

    proposals = args[0]
    ground_truth = args[1]

    result = original_assign(*args, **kwargs)

    assert result.gt_inds.numel() == proposals.shape[0]

    if ground_truth.shape[0] == 0:
        assert result.num_gts == 0
        assert result.max_overlaps is None
        assert torch.all(result.gt_inds == 0).item()
        assert torch.all(result.labels == -1).item()

        result.max_overlaps = proposals.new_zeros(
            proposals.shape[0]
        )

        empty_assignments["count"] += 1

        print(
            "EMPTY_GT_COMPATIBILITY_APPLIED:",
            "proposals=",
            proposals.shape[0],
            flush=True,
        )
    else:
        assert result.num_gts == ground_truth.shape[0]
        assert result.max_overlaps is not None
        assert result.max_overlaps.shape == (
            proposals.shape[0],
        )
        assert torch.isfinite(
            result.max_overlaps
        ).all().item()

    return result

head.bbox_assigner.assign = empty_gt_safe_assign

original_targets = head.get_targets
diagnostics = {}

def consistent_targets(*args, **kwargs):
    result = list(original_targets(*args, **kwargs))

    labels = result[0].reshape(-1)
    query_classes = head.query_labels.reshape(-1)

    weights = result[3]
    flat = weights.clone().reshape(
        -1, weights.shape[-1]
    )

    assert len(labels) == len(query_classes)

    positive = flat[:, 3:6].sum(1) > 0

    mismatch = (
        positive
        & (labels != query_classes)
        & (labels < head.num_classes)
    )

    diagnostics["positive"] = int(positive.sum().item())
    diagnostics["mismatch"] = int(mismatch.sum().item())

    flat[mismatch, 3:6] = 0.0
    result[3] = flat.reshape_as(weights)

    return tuple(result)

head.get_targets = consistent_targets

model.cuda()
training_mode(model)

expected_frozen = {
    "encoders.camera.vtransform.dx",
    "encoders.camera.vtransform.bx",
    "encoders.camera.vtransform.nx",
    "encoders.camera.vtransform.frustum",
}

actual_frozen = {
    name
    for name, param in model.named_parameters()
    if not param.requires_grad
}

assert actual_frozen == expected_frozen, (
    "Unexpected frozen parameters:",
    actual_frozen.symmetric_difference(expected_frozen)
)

total_parameters = sum(
    p.numel() for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    "Total parameters:", total_parameters,
    flush=True
)

print(
    "Trainable parameters:", trainable_parameters,
    flush=True
)

print(
    "Frozen geometric tensors:",
    len(actual_frozen),
    flush=True
)

assert trainable_parameters == 36358814

offset_params = list(head.class_dim_offset.parameters())
offset_ids = {id(p) for p in offset_params}

shared_params = [
    p for p in model.parameters()
    if id(p) not in offset_ids
]

optimizer = torch.optim.AdamW([
    {
        "params": shared_params,
        "lr": LR,
        "weight_decay": 0.01,
    },
    {
        "params": offset_params,
        "lr": LR,
        "weight_decay": 0.0,
    },
])

initial_offset = (
    head.class_dim_offset.weight.detach().clone()
)

print("Optimizer: AdamW", flush=True)
print("Shared LR:", LR, flush=True)
print("Offset LR:", LR, flush=True)
print("Pedestrian NMS exemption: DISABLED", flush=True)
print("Training scope: FULL MODEL", flush=True)


# === Resume verification ===

completed_steps = 0

if args.resume is not None:
    resume_path = Path(args.resume).resolve()

    assert resume_path.parent == OUT, resume_path

    checkpoint = torch.load(
        resume_path, map_location="cpu"
    )

    assert checkpoint["training_type"] == "full_resume_v1"
    assert checkpoint["source_checkpoint"] == str(SOURCE)
    assert checkpoint["schedule_hash"] == schedule_hash
    assert checkpoint["training_indices"] == indices
    assert checkpoint["total_steps"] == STEPS
    assert checkpoint["seed"] == SEED
    assert checkpoint["learning_rate"] == LR
    assert checkpoint["save_every"] == args.save_every

    model.load_state_dict(
        checkpoint["model"], strict=True
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    completed_steps = int(checkpoint["step"])

    assert 0 < completed_steps < args.stop_after

    torch.set_rng_state(
        checkpoint["torch_rng_state"]
    )

    torch.cuda.set_rng_state_all(
        checkpoint["cuda_rng_state"]
    )

    np.random.set_state(
        checkpoint["numpy_rng_state"]
    )

    random.setstate(
        checkpoint["python_rng_state"]
    )

    del checkpoint

    print(
        "RESUME RESTORED:",
        completed_steps,
        flush=True,
    )

print(
    "STARTING FROM STEP:",
    completed_steps + 1,
    flush=True,
)

for step in range(
    completed_steps + 1,
    args.stop_after + 1,
):
    index = indices[step - 1]

    sample = dataset[index]
    assert sample is not None, (
        "Dataset returned None",
        index,
    )

    batch = prepare(sample)

    training_mode(model)
    optimizer.zero_grad(set_to_none=True)

    outputs = model(**batch)

    losses = {}

    for key, value in outputs.items():
        name = key.split("/")[-1]

        if "loss" not in name:
            continue

        if isinstance(value, (tuple, list)):
            value = sum(value)

        assert torch.isfinite(value).all(), name
        losses[name] = value

    required = (
        "loss_heatmap",
        "layer_-1_loss_cls",
        "layer_-1_loss_bbox",
    )

    for name in required:
        assert name in losses, name

    total = sum(losses.values())
    assert torch.isfinite(total).item()

    total.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 35.0
    )

    assert torch.isfinite(grad_norm).item()

    optimizer.step()

    print(
        f"STEP {step}/{STEPS} "
        f"index={index} "
        f"total={total.item():.6f} "
        f"heatmap={losses['loss_heatmap'].item():.6f} "
        f"cls={losses['layer_-1_loss_cls'].item():.6f} "
        f"bbox={losses['layer_-1_loss_bbox'].item():.6f} "
        f"positive={diagnostics['positive']} "
        f"mismatch={diagnostics['mismatch']} "
        f"grad_norm={float(grad_norm):.6f}",
        flush=True,
    )

    if (
        step % args.save_every == 0
        or step == args.stop_after
    ):
        checkpoint_path = OUT / f"step{step}.pth"
        temporary_path = OUT / f"step{step}.pth.tmp"

        assert not checkpoint_path.exists()
        assert not temporary_path.exists()

        payload = {
            "training_type": "full_resume_v1",
            "model": {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            },
            "optimizer": optimizer.state_dict(),
            "step": step,
            "total_steps": STEPS,
            "source_checkpoint": str(SOURCE),
            "schedule_hash": schedule_hash,
            "training_indices": indices,
            "learning_rate": LR,
            "save_every": args.save_every,
            "seed": SEED,
            "pedestrian_nms_exempt": False,
            "mismatch_dim_mask_enabled": True,
            "empty_gt_adapter_enabled": True,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }

        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)

        print(
            "CHECKPOINT SAVED:",
            checkpoint_path,
            flush=True,
        )

head.bbox_assigner.assign = original_assign
head.get_targets = original_targets
dim_handle.remove()

print("TRAINING SEGMENT COMPLETE", flush=True)
print("COMPLETED STEP:", args.stop_after, flush=True)
