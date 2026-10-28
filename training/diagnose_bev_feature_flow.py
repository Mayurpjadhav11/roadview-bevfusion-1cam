#!/usr/bin/env python3
"""Read-only feature-flow diagnostic for the CARLA BEVFusion checkpoint."""

import hashlib
from pathlib import Path

import torch
from mmcv.parallel import DataContainer

from carla_dataset_config import build_dataset
from carla_model_loader import build_carla_model


CHECKPOINT = Path(
    "/outputs/checkpoints/"
    "carla_pilot_20steps_20261028T081055Z/"
    "pilot_last.pth"
)

SCENES = [0, 1, 2]


def unwrap(value):
    return value.data if isinstance(value, DataContainer) else value


def to_gpu(value):
    value = unwrap(value)
    return value.to("cuda").contiguous()


def prepare(sample):
    matrices = [
        "camera2ego",
        "lidar2ego",
        "lidar2camera",
        "lidar2image",
        "camera_intrinsics",
        "camera2lidar",
        "img_aug_matrix",
        "lidar_aug_matrix",
    ]

    inputs = {
        "img": to_gpu(sample["img"]).unsqueeze(0),
        "points": [to_gpu(sample["points"])],
        "metas": [unwrap(sample["metas"])],
    }

    for key in matrices:
        inputs[key] = to_gpu(sample[key]).unsqueeze(0)

    return inputs


def first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value

    if isinstance(value, (list, tuple)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found

    if isinstance(value, dict):
        for item in value.values():
            found = first_tensor(item)
            if found is not None:
                return found

    return None


def digest(tensor):
    data = tensor.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


print("=" * 68)
print("STEP 11.6.6 — BEVFUSION FEATURE FLOW DIAGNOSTIC")
print("=" * 68)

assert CHECKPOINT.is_file()

dataset = build_dataset("val")

torch.manual_seed(42)

model, cfg = build_carla_model(
    pretrained=True,
    device="cuda",
)

checkpoint = torch.load(
    str(CHECKPOINT),
    map_location="cpu",
)

model.load_state_dict(
    checkpoint["model"],
    strict=True,
)

model.eval()

print("\nCheckpoint step:", checkpoint["step"])

head = model.heads["object"]

captured = {}
handles = []


def make_hook(name):
    def hook(module, inputs, output):
        tensor = first_tensor(output)
        if tensor is not None:
            captured[name] = tensor.detach().float().cpu()
    return hook


def make_pre_hook(name):
    def hook(module, inputs):
        tensor = first_tensor(inputs)
        if tensor is not None:
            captured[name] = tensor.detach().float().cpu()
    return hook


# Inspect branches when present in this model version.
camera = model.encoders["camera"]
lidar = model.encoders["lidar"]

if "vtransform" in camera:
    handles.append(
        camera["vtransform"].register_forward_hook(
            make_hook("camera_bev")
        )
    )

if "backbone" in lidar:
    handles.append(
        lidar["backbone"].register_forward_hook(
            make_hook("lidar_backbone")
        )
    )

if model.fuser is not None:
    handles.append(
        model.fuser.register_forward_hook(
            make_hook("fused_bev")
        )
    )

handles.append(
    head.shared_conv.register_forward_pre_hook(
        make_pre_hook("head_input")
    )
)

handles.append(
    head.shared_conv.register_forward_hook(
        make_hook("shared_conv_output")
    )
)

handles.append(
    head.heatmap_head.register_forward_hook(
        make_hook("dense_heatmap_logits")
    )
)

previous = {}

for index in SCENES:
    captured.clear()

    sample = dataset[index]
    inputs = prepare(sample)

    print(f"\n=== VALIDATION SCENE {index:04d} ===")

    print(
        "Image fingerprint:",
        digest(inputs["img"])
    )
    print(
        "LiDAR fingerprint:",
        digest(inputs["points"][0])
    )
    print(
        "LiDAR points:",
        inputs["points"][0].shape[0]
    )

    with torch.no_grad():
        outputs = model(**inputs)

    torch.cuda.synchronize()

    assert "dense_heatmap_logits" in captured
    assert "head_input" in captured

    for name, tensor in captured.items():
        values = tensor.float()

        print(f"\n{name}")
        print("  Shape:", tuple(values.shape))
        print("  Mean:", round(values.mean().item(), 9))
        print("  Std:", round(values.std().item(), 9))
        print(
            "  Nonzero fraction:",
            round((values != 0).float().mean().item(), 9)
        )

        print("  Fingerprint:", digest(values))

        if name == "dense_heatmap_logits":
            probabilities = values.sigmoid()

            spatial_std = probabilities.flatten(2).std(
                dim=-1
            ).mean().item()

            print(
                "  Mean foreground probability:",
                round(probabilities.mean().item(), 9)
            )
            print(
                "  Mean within-class spatial std:",
                round(spatial_std, 9)
            )

        if name in previous:
            earlier = previous[name]

            if earlier.shape == values.shape:
                difference = (values - earlier).abs()

                print(
                    "  Max change vs previous scene:",
                    round(difference.max().item(), 12)
                )
                print(
                    "  Mean change vs previous scene:",
                    round(difference.mean().item(), 12)
                )
            else:
                print(
                    "  Shape changed vs previous scene"
                )

        previous[name] = values.clone()


for handle in handles:
    handle.remove()

print("\n" + "=" * 68)
print("FEATURE FLOW DIAGNOSTIC COMPLETE")
print("=" * 68)
print("No optimizer steps performed.")
print("No checkpoint or source files modified.")
