#!/usr/bin/env python3

"""Verify ImageNet ResNet50 weights against our BEVFusion backbone."""

from pathlib import Path
import torch
import yaml

from mmcv import Config
from torchpack.utils.config import configs
from mmdet3d.utils import recursive_eval
from mmdet3d.models import build_model

BASE = Path(
    "/workspace/bevfusion/configs/nuscenes/det/"
    "transfusion/secfpn/camera+lidar/"
    "resnet50/convfuser.yaml"
)

OVERRIDES = Path(
    "/workspace/project/configs/one_camera/"
    "carla_resnet50_overrides.yaml"
)

CHECKPOINT = Path(
    "/outputs/pretrained/resnet50-0676ba61.pth"
)


def apply_overrides(target, changes):
    for key, value in changes.items():
        if key not in target:
            raise KeyError(f"Unknown configuration key: {key}")

        if isinstance(value, dict):
            apply_overrides(target[key], value)
        else:
            target[key] = value


print("=" * 60)
print("STEP 11.4.6 — RESNET50 CHECKPOINT COMPATIBILITY")
print("=" * 60)

assert BASE.is_file()
assert OVERRIDES.is_file()
assert CHECKPOINT.is_file()

print("\n=== 1. LOAD CARLA MODEL CONFIGURATION ===")

configs.load(str(BASE), recursive=True)

with OVERRIDES.open() as f:
    apply_overrides(configs, yaml.safe_load(f))

cfg = Config(recursive_eval(configs), filename=str(BASE))

assert cfg.model.encoders.camera.backbone.type == "ResNet"
assert cfg.model.encoders.camera.backbone.depth == 50
assert cfg.model.encoders.camera.backbone.init_cfg is None
assert cfg.model.heads.object.num_classes == 6

print("Architecture: ResNet50 BEVFusion")
print("Detection classes: 6")


print("\n=== 2. BUILD MODEL ON CPU ===")

model = build_model(cfg.model)
model.cpu()

backbone = model.encoders["camera"]["backbone"]

print("Camera backbone:", type(backbone).__name__)


print("\n=== 3. INSPECT DOWNLOADED CHECKPOINT ===")

# The checkpoint comes from the verified official
# PyTorch download. Load only trusted checkpoints.
checkpoint = torch.load(
    str(CHECKPOINT),
    map_location="cpu",
)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

assert isinstance(checkpoint, dict)

print("Checkpoint tensor entries:", len(checkpoint))


print("\n=== 4. COMPARE PARAMETER NAMES AND SHAPES ===")

target = backbone.state_dict()

matched = {}
unexpected = []
shape_mismatches = []

for key, value in checkpoint.items():

    if key not in target:
        unexpected.append(key)
        continue

    if not isinstance(value, torch.Tensor):
        shape_mismatches.append((key, "not a tensor"))
        continue

    if value.shape != target[key].shape:
        shape_mismatches.append(
            (key, tuple(value.shape), tuple(target[key].shape))
        )
        continue

    matched[key] = value

missing = sorted(set(target) - set(matched))

print("Backbone state entries:", len(target))
print("Matching entries:", len(matched))
print("Missing entries:", len(missing))
print("Unused checkpoint entries:", len(unexpected))
print("Shape mismatches:", len(shape_mismatches))

print("\nUnused entries:", unexpected[:15])
print("Missing entries:", missing[:15])
print("Shape mismatches:", shape_mismatches[:10])


print("\n=== 5. MEASURE PARAMETER COVERAGE ===")

parameters = dict(backbone.named_parameters())

total_elements = sum(
    p.numel() for p in parameters.values()
)

matched_elements = sum(
    p.numel()
    for name, p in parameters.items()
    if name in matched
)

coverage = matched_elements / total_elements

print("Backbone parameter elements:", total_elements)
print("Matched parameter elements:", matched_elements)
print("Parameter coverage: {:.2f}%".format(coverage * 100))

assert coverage >= 0.98, (
    "Insufficient backbone weight compatibility"
)


print("\n=== 6. LOAD WEIGHTS INTO RESNET50 ===")

before = backbone.conv1.weight.detach().clone()

result = backbone.load_state_dict(
    matched,
    strict=False,
)

print("Missing after load:", len(result.missing_keys))
print("Unexpected after load:", len(result.unexpected_keys))

assert "conv1.weight" in matched

assert torch.equal(
    backbone.conv1.weight.detach(),
    matched["conv1.weight"],
)

assert not torch.equal(
    before,
    backbone.conv1.weight.detach(),
), "Backbone weights did not change"

print("First convolution weights loaded correctly")


print("\n" + "=" * 60)
print("PRETRAINED RESNET50 COMPATIBILITY PASSED")
print("=" * 60)

print("Checkpoint:", CHECKPOINT.name)
print("Backbone coverage: {:.2f}%".format(coverage * 100))
print("Model remains on CPU")
print("No training performed")
print("No checkpoint was modified")
print("No model was saved")
