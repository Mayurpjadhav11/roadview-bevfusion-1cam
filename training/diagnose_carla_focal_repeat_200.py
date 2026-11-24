from pathlib import Path
import numpy as np
import torch
from mmcv.parallel import DataContainer

from mmdet3d.models.heads.bbox.transfusion import (
    gaussian_radius,
    draw_heatmap_gaussian,
)
from carla_dataset_config import build_dataset
from carla_model_loader import build_carla_model

ROOT = Path("/outputs/checkpoints")
CHECKPOINTS = [
    ("SOURCE", ROOT /
     "hard_ped_center_fixed_20261028T114303Z/step40.pth"),
    ("NOAUX20", ROOT /
     "carla_multiscene_noaux_20_20261028T134838Z/step20.pth"),
    ("REPEAT20", Path("/outputs/checkpoints/carla_multiscene_repeat_200_20261028T141609Z/step20.pth")),
    ("REPEAT50", Path("/outputs/checkpoints/carla_multiscene_repeat_200_20261028T141609Z/step50.pth")),
    ("REPEAT100", Path("/outputs/checkpoints/carla_multiscene_repeat_200_20261028T141609Z/step100.pth")),
    ("REPEAT200", Path("/outputs/checkpoints/carla_multiscene_repeat_200_20261028T141609Z/step200.pth")),
]
INDICES = [0, 222, 4162, 2507]

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
    return x.cuda().float().contiguous()

def prepare(sample):
    data = {
        "img": gpu(sample["img"]).unsqueeze(0),
        "points": [gpu(sample["points"])],
        "metas": [unwrap(sample["metas"])],
    }
    for key in MATRICES:
        data[key] = gpu(sample[key]).unsqueeze(0)
    return data

torch.manual_seed(42)

dataset = build_dataset("train")
samples = []

for index in INDICES:
    sample = dataset[index]
    assert sample is not None
    samples.append((
        index,
        dataset.data_infos[index]["token"],
        sample,
        prepare(sample),
    ))

model, _ = build_carla_model(
    pretrained=False, device="cpu"
)

head = model.heads["object"]
head.class_dim_offset = torch.nn.Embedding(
    head.num_classes, 3
)
model.cuda().eval()

captured = {}

def capture_logits(module, args, output):
    captured["logits"] = output.detach().float().cpu()[0]

hook = head.heatmap_head.register_forward_hook(
    capture_logits
)

cfg = head.train_cfg
pc = cfg["point_cloud_range"]
vx = cfg["voxel_size"]
factor = cfg["out_size_factor"]
dx = vx[0] * factor
dy = vx[1] * factor

def build_target(sample, shape):
    target = torch.zeros(shape, dtype=torch.float32)
    boxes = unwrap(sample["gt_bboxes_3d"])
    labels = torch.as_tensor(
        unwrap(sample["gt_labels_3d"])
    ).cpu().long()

    centers = boxes.gravity_center.cpu()
    box_data = boxes.tensor.cpu()

    valid_positions = []

    for i in range(len(labels)):
        cls = int(labels[i])

        gx = int((centers[i, 0].item() - pc[0]) / dx)
        gy = int((centers[i, 1].item() - pc[1]) / dy)

        if not (
            0 <= cls < shape[0]
            and 0 <= gx < shape[1]
            and 0 <= gy < shape[2]
        ):
            continue

        # IMPORTANT FIX: keep dimensions as tensors.
        width = box_data[i, 3] / dx
        length = box_data[i, 4] / dy

        if width.item() <= 0 or length.item() <= 0:
            continue

        radius = gaussian_radius(
            (length, width),
            min_overlap=cfg["gaussian_overlap"],
        )
        radius = max(
            cfg["min_radius"],
            int(radius),
        )

        # Current CARLA target coordinate convention.
        draw_heatmap_gaussian(
            target[cls],
            torch.tensor([gy, gx], dtype=torch.int32),
            radius,
        )

        valid_positions.append((cls, gx, gy))

    return target, len(labels), valid_positions

print("=== STEP 11.9.13: CORRECTED FOCAL AUDIT ===")
print("Tensor dimension fix: enabled")
print("No parameter updates")

for model_name, path in CHECKPOINTS:
    checkpoint = torch.load(
        path, map_location="cpu"
    )
    model.load_state_dict(
        checkpoint["model"], strict=True
    )
    model.eval()

    for index, token, sample, inputs in samples:
        captured.clear()

        with torch.no_grad():
            _ = model(**inputs)

        logits = captured["logits"]
        target, gt_count, positions = build_target(
            sample, logits.shape
        )

        peaks = target.eq(1)
        background = target.lt(1)
        peak_count = int(peaks.sum())
        normalization = max(peak_count, 1)

        pred = logits.sigmoid().clamp(
            min=1e-4, max=1 - 1e-4
        )

        positive = (
            -pred.log()
            * (1 - pred).pow(2)
            * peaks.float()
        ).sum() / normalization

        negative = (
            -(1 - pred).log()
            * pred.pow(2)
            * (1 - target).pow(4)
            * background.float()
        ).sum() / normalization

        with torch.no_grad():
            configured_loss = head.loss_heatmap(
                pred.unsqueeze(0).cuda(),
                target.unsqueeze(0).cuda(),
                avg_factor=normalization,
            )

        gt_probs = [
            pred[cls, gx, gy].item()
            for cls, gx, gy in positions
        ]

        gt_mean = (
            float(np.mean(gt_probs))
            if gt_probs else float("nan")
        )

        total = positive + negative

        print(
            f"MODEL={model_name} "
            f"TOKEN={token} "
            f"GT={gt_count} "
            f"VALID={len(positions)} "
            f"PEAKS={peak_count} "
            f"GT_MEAN={gt_mean:.6f} "
            f"POS_LOSS={positive.item():.5f} "
            f"NEG_LOSS={negative.item():.5f} "
            f"SUM={total.item():.5f} "
            f"CONFIG_LOSS={configured_loss.item():.5f}",
            flush=True
        )

hook.remove()

print("STEP 11.9.13 COMPLETE")
print("No checkpoints or source code modified.")
