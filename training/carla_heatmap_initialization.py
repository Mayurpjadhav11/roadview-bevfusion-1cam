#!/usr/bin/env python3
"""Optional heatmap initialization for CARLA BEVFusion."""

import math
import torch

from carla_model_loader import build_carla_model


def apply_heatmap_prior(model, probability=0.01):
    """
    Initialize the final dense heatmap biases.

    This changes only the final heatmap bias.
    It does not alter other network parameters.
    """

    if not 0.0 < probability < 1.0:
        raise ValueError(
            "Foreground probability must be between 0 and 1"
        )

    head = model.heads["object"]

    parameters = dict(
        head.heatmap_head.named_parameters()
    )

    assert "1.bias" in parameters, (
        "Unexpected dense heatmap head structure"
    )

    bias = parameters["1.bias"]

    assert bias.numel() == 6, (
        "Expected one bias per CARLA class"
    )

    logit = math.log(
        probability / (1.0 - probability)
    )

    with torch.no_grad():
        bias.fill_(logit)

    assert torch.allclose(
        bias.detach().sigmoid(),
        torch.full_like(bias, probability),
        atol=1e-6,
    )

    return logit


if __name__ == "__main__":

    print("=" * 60)
    print("STEP 11.7.1 — OPTIONAL HEATMAP INITIALIZATION")
    print("=" * 60)

    torch.manual_seed(42)

    model, _ = build_carla_model(
        pretrained=True,
        device="cpu",
    )

    # Preserve the complete starting state.
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    print("\n=== APPLY 1% FOREGROUND PRIOR ===")

    logit = apply_heatmap_prior(
        model,
        probability=0.01,
    )

    after = model.state_dict()

    changed = [
        name
        for name in before
        if not torch.equal(before[name], after[name])
    ]

    print("Target probability:", 0.01)
    print("Target bias logit:", logit)
    print("Changed state entries:", changed)

    expected = "heads.object.heatmap_head.1.bias"

    assert changed == [expected], (
        f"Unexpected changed entries: {changed}"
    )

    print("\n=== VERIFY UNCHANGED COMPONENTS ===")

    for component in [
        "encoders.camera",
        "encoders.lidar",
        "fuser",
        "decoder",
    ]:
        unchanged = all(
            torch.equal(value, after[name])
            for name, value in before.items()
            if name.startswith(component)
        )

        assert unchanged, component

        print(component, ": UNCHANGED")

    print("\n" + "=" * 60)
    print("OPTIONAL HEATMAP INITIALIZER PASSED")
    print("=" * 60)

    print("Only the six heatmap biases were modified.")
    print("No training performed.")
    print("No checkpoint saved.")
    print("NVIDIA source unchanged.")
