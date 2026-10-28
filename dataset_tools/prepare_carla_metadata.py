
from pathlib import Path
import copy
import pickle
import numpy as np

# Dataset is mounted at /dataset inside Docker.
DATASET_ROOT = Path("/dataset")

# Write generated metadata outside the Git repository.
OUTPUT_DIR = Path("/outputs/experiments/carla_metadata")


def convert_path(raw_path):
    """Convert CARLA Windows paths into Docker Linux paths."""
    normalized = raw_path.replace("\\", "/")

    if normalized.startswith("dataset/"):
        normalized = normalized[len("dataset/"):]

    if normalized.startswith("/"):
        raise ValueError("Unexpected absolute dataset path")

    path = DATASET_ROOT / normalized

    if not path.is_file():
        raise FileNotFoundError(path)

    return str(path)


def convert_sample(info):
    """Adapt one CARLA sample for the existing BEVFusion loader."""
    info = copy.deepcopy(info)

    # 1. Correct LiDAR file path.
    info["lidar_path"] = convert_path(info["lidar_path"])

    # 2. Correct camera paths and intrinsic field name.
    for camera in info["cams"].values():
        camera["data_path"] = convert_path(camera["data_path"])

        if "camera_intrinsics" not in camera:
            camera["camera_intrinsics"] = camera["cam_intrinsic"]

    # 3. Correct geometric-center Z to bottom-center Z.
    boxes = np.asarray(info["gt_boxes"]).copy()

    if len(boxes):
        boxes[:, 2] -= boxes[:, 5] / 2.0

    info["gt_boxes"] = boxes

    # 4. Preserve sweeps, but normalize any sweep paths.
    for sweep in info["sweeps"]:
        if "data_path" in sweep:
            sweep["data_path"] = convert_path(sweep["data_path"])

    return info


def convert_split(split):
    input_path = DATASET_ROOT / "bevfusion" / (
        "carla_infos_{}.pkl".format(split)
    )

    output_path = OUTPUT_DIR / (
        "carla_infos_{}.pkl".format(split)
    )

    print("Loading:", input_path)

    with open(input_path, "rb") as f:
        original = pickle.load(f)

    converted = {
        "infos": [],
        "metadata": copy.deepcopy(original["metadata"]),
    }

    total = len(original["infos"])

    for i, info in enumerate(original["infos"]):
        converted["infos"].append(convert_sample(info))

        if (i + 1) % 2000 == 0:
            print("Processed {}/{}".format(i + 1, total))

    # Validate the output structure before writing.
    assert len(converted["infos"]) == total

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(converted, f, protocol=4)

    print("Saved:", output_path)
    print("Samples:", total)


if __name__ == "__main__":
    convert_split("train")
    convert_split("val")
    print("Metadata adaptation completed.")
