# Step 11.8 — CARLA Single-Scene BEVFusion Overfitting

Date: 2026-10-28

## Status

PASS — Five distinct, class-correct 3D detections.

## Configuration

- Detector: BEVFusion with TransFusion detection head.
- Sensors: one CARLA camera and LiDAR.
- Classes: car, truck, bus, motorcycle, bicycle, pedestrian.
- Training scene: CARLA training sample index 0.
- Ground truth: one car and four pedestrians.
- Closest pedestrian pair: approximately 0.464 m apart.

## Successful Experimental Changes

1. Class-conditioned dimension offsets.
2. Pedestrian-specific exemption from 3x3 heatmap suppression.
3. Corrected hard-pedestrian XY center supervision.
4. Frozen BatchNorm during single-scene optimization.
5. Class-aware, one-to-one Hungarian 3D IoU evaluation.

The class dimension correction is implemented through a forward
hook, and the pedestrian NMS exemption is applied through a
runtime forward patch. These must also be installed at inference.

## Selected Checkpoint

Docker path:

/outputs/checkpoints/hard_ped_center_fixed_20261028T114303Z/step40.pth

SHA-256:

c87bd9307ee79dbffd0c2352de57652025a2bb35d0182165cbb237a7798de2f0

The model checkpoint is stored outside Git.

## Verified Results

| Object | Class | Rotated 3D IoU | Confidence |
|---|---|---:|---:|
| GT0 | car | 0.8458 | 0.08545 |
| GT1 | pedestrian | 0.3627 | 0.13456 |
| GT2 | pedestrian | 0.3729 | 0.00382 |
| GT3 | pedestrian | 0.4143 | 0.13240 |
| GT4 | pedestrian | 0.3143 | 0.14179 |

- Mean 3D IoU: 0.46201
- Five distinct matches at IoU >= 0.25
- Car IoU >= 0.50
- Fresh evaluation process: PASS

## Reproduce Evaluation

Within the bevfusion-train container:

    cd /workspace/bevfusion
    export PYTHONPATH=/workspace/project/training:$PYTHONPATH
    python /workspace/project/training/evaluate_hard_ped_center_fixed.py

This evaluator compares the source checkpoint and all four
hard-center fine-tuned checkpoints.

## Limitations

- Only one training scene has been verified.
- Validation/generalization has not been demonstrated.
- GT2 has extremely low confidence (0.00382).
- Evaluation uses class-aware matching without a production
  confidence threshold.
- False positives, calibration and inference stability need testing.
- TensorRT/DRIVE AGX Orin deployment has not been validated.
- The Docker volume containing upstream BEVFusion is not itself
  a Git working tree; its source provenance must be tracked
  separately.

## Next Milestone

Step 11.9 — Multi-scene training and validation, including
confidence calibration, false positives and detection robustness.
