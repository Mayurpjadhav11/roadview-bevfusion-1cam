#!/usr/bin/env python3

"""Reusable ResNet50 BEVFusion model loader for CARLA."""

from pathlib import Path

import torch
import yaml

from mmcv import Config
from torchpack.utils.config import configs

from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


BASE_CONFIG = Path(
    "/workspace/bevfusion/configs/nuscenes/det/"
    "transfusion/secfpn/camera+lidar/"
    "resnet50/convfuser.yaml"
)

CARLA_CONFIG = Path(
    "/workspace/project/configs/one_camera/"
    "carla_resnet50_overrides.yaml"
)

RESNET50_WEIGHTS = Path(
    "/outputs/pretrained/resnet50-0676ba61.pth"
)


def apply_overrides(target, changes, prefix=""):
    for key, value in changes.items():

        full_name = f"{prefix}.{key}" if prefix else key

        if key not in target:
            raise KeyError(
                f"Unknown configuration key: {full_name}"
            )

        if isinstance(value, dict):
            apply_overrides(
                target[key],
                value,
                full_name,
            )
        else:
            target[key] = value


def load_carla_config():

    assert BASE_CONFIG.is_file()
    assert CARLA_CONFIG.is_file()

    configs.load(str(BASE_CONFIG), recursive=True)

    with CARLA_CONFIG.open("r") as f:
        overrides = yaml.safe_load(f)

    apply_overrides(configs, overrides)

    cfg = Config(
        recursive_eval(configs),
        filename=str(BASE_CONFIG),
    )

    assert cfg.model.encoders.camera.backbone.type == "ResNet"
    assert cfg.model.encoders.camera.backbone.depth == 50
    assert cfg.model.encoders.camera.backbone.init_cfg is None
    assert cfg.model.heads.object.num_classes == 6

    return cfg


def build_carla_model(pretrained=True, device="cpu"):

    cfg = load_carla_config()

    # Build the complete CARLA BEVFusion architecture.
    model = build_model(cfg.model)

    if pretrained:

        assert RESNET50_WEIGHTS.is_file(), (
            f"Missing checkpoint: {RESNET50_WEIGHTS}"
        )

        checkpoint = torch.load(
            str(RESNET50_WEIGHTS),
            map_location="cpu",
        )

        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        backbone = model.encoders["camera"]["backbone"]
        target_state = backbone.state_dict()

        # Load only matching ResNet50 entries.
        # The ImageNet fc.weight and fc.bias are excluded.
        matched = {
            key: value
            for key, value in checkpoint.items()
            if key in target_state
            and isinstance(value, torch.Tensor)
            and value.shape == target_state[key].shape
        }

        parameters = dict(backbone.named_parameters())

        matched_parameters = sum(
            parameter.numel()
            for name, parameter in parameters.items()
            if name in matched
        )

        total_parameters = sum(
            parameter.numel()
            for parameter in parameters.values()
        )

        coverage = (
            matched_parameters / total_parameters
        )

        assert coverage >= 0.999, (
            f"Insufficient ResNet50 coverage: {coverage:.2%}"
        )

        result = backbone.load_state_dict(
            matched,
            strict=False,
        )

        # BatchNorm num_batches_tracked buffers can be
        # absent in compatible older checkpoints.
        unexpected_missing = [
            key
            for key in result.missing_keys
            if not key.endswith("num_batches_tracked")
        ]

        assert not unexpected_missing, unexpected_missing
        assert not result.unexpected_keys

        # Verify the actual model parameters were updated.
        assert torch.equal(
            backbone.conv1.weight.detach().cpu(),
            checkpoint["conv1.weight"],
        )

        print(
            "ResNet50 pretrained parameter coverage: "
            f"{coverage:.2%}"
        )

    model = model.to(device)

    return model, cfg


if __name__ == "__main__":

    print("=" * 55)
    print("STEP 11.4.7 — CARLA PRETRAINED MODEL LOADER")
    print("=" * 55)

    model, cfg = build_carla_model(
        pretrained=True,
        device="cpu",
    )

    backbone = model.encoders["camera"]["backbone"]

    print("\n=== MODEL VERIFICATION ===")

    print("Model:", type(model).__name__)
    print("Camera backbone:", type(backbone).__name__)
    print("Camera depth:", cfg.model.encoders.camera.backbone.depth)
    print("Detection classes:", cfg.model.heads.object.num_classes)
    print("Device:", next(model.parameters()).device)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Total parameters:", f"{total_parameters:,}")

    assert cfg.model.heads.object.num_classes == 6
    assert total_parameters == 37355669

    print("\n" + "=" * 55)
    print("PRETRAINED CARLA MODEL LOADER PASSED")
    print("=" * 55)

    print("ResNet50 pretrained: YES")
    print("Other BEVFusion modules pretrained: NO")
    print("No training performed.")
    print("No model weights saved.")
