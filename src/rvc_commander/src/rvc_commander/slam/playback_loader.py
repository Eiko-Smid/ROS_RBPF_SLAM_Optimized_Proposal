#!/usr/bin/env python3

import csv
import json
import numpy as np
from typing import List
from rvc_commander.slam.scan_match_playback_def import (
    StepData,
    SensorParam,
    OccupancyParam,
    MapData,
)


def load_map(map_npy_path: str, map_meta_path: str) -> MapData:
    # Load map array
    log_odds_map = np.load(map_npy_path)

    # Load metadata
    meta = {}
    with open(map_meta_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for key, value in reader:
            meta[key] = float(value)

    return MapData(
        min_distance_to_border=meta["min_distance_to_border"],
        log_odds_map=log_odds_map,
        sensor_param=SensorParam(
            min_sensor_range=meta["min_sensor_range"],
            max_sensor_range=meta["max_sensor_range"],
        ),
        occupancy_param=OccupancyParam(
            prior_probability=meta["prior_probability"],
            increasing_probability=meta["increasing_probability"],
            decreasing_probability=meta["decreasing_probability"],
            min_log_odds=meta["min_log_odds"],
            max_log_odds=meta["max_log_odds"],
        ),
    )


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


def load_playback_dataset(base_path_prefix: str):
    """
    Example:
    base_path_prefix = "/path/to/123456_python_playback_data"
    """

    map_data = load_map(
        map_npy_path=base_path_prefix + "_map.npy",
        map_meta_path=base_path_prefix + "_map_meta.csv",
    )

    steps = load_steps(
        steps_csv_path=base_path_prefix + "_steps.csv",
        scans_jsonl_path=base_path_prefix + "_scans.jsonl",
    )

    return map_data, steps


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