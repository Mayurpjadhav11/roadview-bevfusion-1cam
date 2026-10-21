# Roadview Single-Camera BEVFusion

## Objective
Adapt the pretrained six-camera + LiDAR BEVFusion model
for one camera + one LiDAR.

## Training
Hardware: Roadview MIFCOM
Framework: PyTorch
Model: BEVFusion ResNet50

## Deployment
Hardware: NVIDIA DRIVE AGX Orin
Framework: CUDA and TensorRT
ONNX: Opset 13

## ROS
ROS 1 integration initially if required.
ROS 2 integration planned.

## Dataset
Stored separately at:
/mnt/DATA_16Tb/Simulated_DATASET/dataset

## Target
Real-time 3D object detection at approximately 10 Hz.
