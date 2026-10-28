#!/usr/bin/env python3
"""Check BEVFusion training-loss computation on one CARLA sample."""

from pathlib import Path
import torch
import yaml

from mmcv import Config
from mmcv.parallel import DataContainer
from torchpack.utils.config import configs
from mmdet3d.utils import recursive_eval
from mmdet3d.models import build_model

from carla_dataset_config import build_dataset

BASE = Path(
    "/workspace/bevfusion/configs/nuscenes/det/"
    "transfusion/secfpn/camera+lidar/resnet50/convfuser.yaml"
)

OVERRIDES = Path(
    "/workspace/project/configs/one_camera/"
    "carla_resnet50_overrides.yaml"
)

def apply_overrides(target, changes):
    for key, value in changes.items():
        if key not in target:
            raise KeyError(f"Unknown config key: {key}")
        if isinstance(value, dict):
            apply_overrides(target[key], value)
        else:
            target[key] = value

def unwrap(value):
    return value.data if isinstance(value, DataContainer) else value

def gpu_tensor(value, dtype=torch.float32):
    value = unwrap(value)
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.to("cuda", dtype=dtype).contiguous()

print("=" * 55)
print("STEP 11.3.5 — CARLA TRAINING LOSS SMOKE TEST")
print("=" * 55)

assert torch.cuda.is_available()

print("\n=== 1. LOAD CONFIGURATION ===")
configs.load(str(BASE), recursive=True)

with OVERRIDES.open() as f:
    apply_overrides(configs, yaml.safe_load(f))

cfg = Config(recursive_eval(configs), filename=str(BASE))
assert cfg.model.heads.object.num_classes == 6
assert cfg.model.encoders.camera.backbone.init_cfg is None

print("ResNet50 configuration: PASSED")
print("Detection classes: 6")

print("\n=== 2. LOAD ANNOTATED CARLA SAMPLE ===")
dataset = build_dataset("val")
sample = dataset[0]

img = gpu_tensor(sample["img"]).unsqueeze(0)
points = [gpu_tensor(sample["points"])]

matrix_keys = [
    "camera2ego",
    "lidar2ego",
    "lidar2camera",
    "lidar2image",
    "camera_intrinsics",
    "camera2lidar",
    "img_aug_matrix",
    "lidar_aug_matrix",
]

matrices = {
    key: gpu_tensor(sample[key]).unsqueeze(0)
    for key in matrix_keys
}

metadata = unwrap(sample["metas"])
assert isinstance(metadata, dict)
metas = [metadata]

boxes = unwrap(sample["gt_bboxes_3d"])
labels = gpu_tensor(
    sample["gt_labels_3d"], dtype=torch.long
)

assert img.shape == (1, 1, 3, 256, 704)
assert points[0].shape[1] == 5
assert len(boxes) == len(labels)
assert len(labels) > 0
assert int(labels.min()) >= 0
assert int(labels.max()) < 6

print("Image:", tuple(img.shape))
print("LiDAR:", tuple(points[0].shape))
print("Ground-truth boxes:", tuple(boxes.tensor.shape))
print("Ground-truth labels:", labels.cpu().tolist())

# Move the annotated 3D boxes onto the same device
# as the network.
gt_bboxes_3d = [boxes.to("cuda")]
gt_labels_3d = [labels]

print("\n=== 3. BUILD MODEL ===")
model = build_model(cfg.model).cuda()
model.train()

print("Model:", type(model).__name__)
print("Device:", next(model.parameters()).device)

torch.cuda.reset_peak_memory_stats()

print("\n=== 4. COMPUTE TRAINING LOSSES ===")

# Training mode activates the loss computation branch.
# no_grad avoids storing backward activations for this test.
with torch.enable_grad():
    outputs = model(
        img=img,
        points=points,
        camera2ego=matrices["camera2ego"],
        lidar2ego=matrices["lidar2ego"],
        lidar2camera=matrices["lidar2camera"],
        lidar2image=matrices["lidar2image"],
        camera_intrinsics=matrices["camera_intrinsics"],
        camera2lidar=matrices["camera2lidar"],
        img_aug_matrix=matrices["img_aug_matrix"],
        lidar_aug_matrix=matrices["lidar_aug_matrix"],
        metas=metas,
        gt_bboxes_3d=gt_bboxes_3d,
        gt_labels_3d=gt_labels_3d,
    )

assert isinstance(outputs, dict), type(outputs)

print("Returned loss/statistic keys:", len(outputs))

loss_count = 0

for name, value in outputs.items():
    if isinstance(value, (tuple, list)):
        values = list(value)
    else:
        values = [value]

    scalars = []

    for item in values:
        assert isinstance(item, torch.Tensor), (
            name, type(item)
        )

        assert torch.isfinite(item).all(), (
            f"Non-finite value in {name}"
        )

        scalars.append(float(item.detach().mean().cpu()))

    print(f"{name}: {scalars}")

    # With torch.no_grad(), BEVFusion returns computed
    # losses under "stats/" because requires_grad=False.
    # Accept those losses for this loss-only smoke test.
    if (
        name.startswith(("loss/", "stats/"))
        and "loss" in name
    ):
        loss_count += 1

assert loss_count > 0, "No training losses were returned"

print("\n=== BACKWARD PROPAGATION TEST ===")

loss_terms = []

for name, value in outputs.items():

    if not name.startswith("loss/"):
        continue

    values = value if isinstance(value, (list, tuple)) else [value]

    for term in values:
        assert isinstance(term, torch.Tensor)
        assert term.requires_grad, name
        assert torch.isfinite(term).all(), name

        loss_terms.append(term.mean())

assert len(loss_terms) > 0, "No differentiable losses found"

total_loss = torch.stack(loss_terms).sum()

print("Differentiable loss terms:", len(loss_terms))
print("Total loss:", float(total_loss.detach().cpu()))

model.zero_grad(set_to_none=True)

print("Running backward pass...")

total_loss.backward()

torch.cuda.synchronize()

print("Backward computation: COMPLETED")


print("\n=== CHECK GRADIENT FLOW ===")

components = {
    "Camera ResNet50": model.encoders["camera"]["backbone"],
    "Camera neck": model.encoders["camera"]["neck"],
    "DepthLSSTransform": model.encoders["camera"]["vtransform"],
    "LiDAR SparseEncoder": model.encoders["lidar"]["backbone"],
    "BEV ConvFuser": model.fuser,
    "TransFusionHead": model.heads["object"],
}

for name, component in components.items():

    gradients = [
        parameter.grad
        for parameter in component.parameters()
        if parameter.grad is not None
    ]

    assert len(gradients) > 0, (
        f"No gradients in {name}"
    )

    for grad in gradients:
        assert torch.isfinite(grad).all(), (
            f"Non-finite gradients in {name}"
        )

    gradient_sum = sum(
        float(grad.detach().abs().sum().item())
        for grad in gradients
    )

    print(
        "{}: tensors={} | absolute gradient sum={:.6f}".format(
            name,
            len(gradients),
            gradient_sum,
        )
    )

    assert gradient_sum > 0, (
        f"Zero gradients in {name}"
    )

print("\nALL CHECKED COMPONENTS HAVE FINITE, NONZERO GRADIENTS")


print("\n=== 5. GPU MEMORY ===")
print(
    "Peak allocated memory: {:.1f} MiB".format(
        torch.cuda.max_memory_allocated() / 1024**2
    )
)

print("\n" + "=" * 55)
print("CARLA BACKWARD PASS PASSED")
print("=" * 55)

print("Loss tensors are finite.")
print("One backward pass was performed.")
print("No weights were updated.")
print("No training was started.")
