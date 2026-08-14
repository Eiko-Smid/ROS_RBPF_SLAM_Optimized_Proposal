import csv
import json
import numpy as np
from typing import List, Optional, Tuple

from .playback_defs import StepData, ExperimentParams


def load_scans(scans_jsonl_path: str):
    scans = {}

    with open(scans_jsonl_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            scans[entry["step_id"]] = entry["scan"]

    return scans


def load_steps(steps_csv_path: str, scans_jsonl_path: str) -> List[StepData]:
    scans = load_scans(scans_jsonl_path)
    steps = []

    with open(steps_csv_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            step_id = int(row["step_id"])

            step = StepData(
                t=float(row["t"]),
                dl=float(row["dl"]),
                dr=float(row["dr"]),
                scan=[tuple(x) for x in scans[step_id]],
                true_pose=(
                    float(row["true_pose_x"]),
                    float(row["true_pose_y"]),
                    float(row["true_pose_yaw"]),
                ),
            )
            steps.append(step)

    return steps


def load_playback_dataset(base_path_prefix: str, n_steps: Optional[int] = None) -> List[StepData]:
    """
    Example:
    base_path_prefix = "/path/to/123456_python_playback_data"
    """
    steps = load_steps(
        steps_csv_path=base_path_prefix + "_steps.csv",
        scans_jsonl_path=base_path_prefix + "_scans.jsonl",
    )

    if n_steps is None:
        return steps

    n_steps = int(n_steps)
    if n_steps <= 0:
        return []

    actual_steps = len(steps)
    if actual_steps < n_steps:
        print(
            f"WARNING: requested n_steps={n_steps}, but dataset only has {actual_steps} steps. "
            "Returning all available steps."
        )
        return steps

    steps = steps[:n_steps]

    return steps


def main():
    print("THis is a test")

    StepData(
        t=0.0,
        dl=0.0,
        dr=0.0,
        scan=[(1.0, 0.0), (1.5, 1.57)],
        true_pose=(0.0, 0.0, 0.0),
    )


if __name__ == "__main__":
    main()