
from dataclasses import dataclass
from typing import List, Optional, Tuple

import os
import csv
import json
import numpy as np


@dataclass
class RawScan:
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float
    ranges: np.ndarray


@dataclass
class PlaybackStep:
    step_id: int
    t: float
    t_ros: float
    dl: float
    dr: float
    true_pose: Tuple[float, float, float]
    scan: RawScan


@dataclass
class PlaybackData:
    metadata: dict
    steps: List[PlaybackStep]


class PlaybackLoader:
    def load(
        self,
        file_suffix,
        filedir,
        n_steps: Optional[int] = None,
        ensure_start_pose: bool = False,
        prompt_for_missing_start_pose: bool = False,
    ) -> PlaybackData:
        # Define paths
        self.meta_path = os.path.join(filedir, f"{file_suffix}_meta.json")
        self.scans_path = os.path.join(filedir, f"{file_suffix}_scans.jsonl")
        self.steps_path = os.path.join(filedir, f"{file_suffix}_steps.csv")

        if n_steps is not None and n_steps < 0:
            raise ValueError(f"n_steps must be >= 0 or None, got {n_steps}")

        metadata = self._load_metadata()
        if ensure_start_pose:
            metadata["robot_start_pose"] = self._resolve_robot_start_pose(
                metadata=metadata,
                prompt_missing=prompt_for_missing_start_pose,
            )
        step_ids = self._load_step_ids(n_steps=n_steps)
        scan_dict = self._load_scans(step_ids=step_ids)
        steps = self._load_steps(scan_dict, n_steps=n_steps)

        return PlaybackData(metadata=metadata, steps=steps)


    def _load_metadata(self):
        with open(self.meta_path, "r") as f:
            return json.load(f)


    @staticmethod
    def _coerce_pose_tuple(value) -> Optional[Tuple[float, float, float]]:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return None

        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            return None


    @staticmethod
    def _prompt_start_pose_fallback() -> Tuple[float, float, float]:
        print("robot_start_pose was not found in playback meta data.")
        print("Choose how to proceed:")
        print("1) Manually define start pose")
        print("2) Use zero pose (0.0, 0.0, 0.0)")
        print("3) Exit")

        while True:
            choice = input("Enter choice (1/2/3): ").strip()

            if choice == "1":
                while True:
                    raw_pose = input("Enter start pose as: x y yaw\n> ").strip()
                    parts = raw_pose.split()

                    if len(parts) != 3:
                        print("Invalid input. Please provide exactly 3 numbers: x y yaw")
                        continue

                    try:
                        x, y, yaw = (float(parts[0]), float(parts[1]), float(parts[2]))
                        return (x, y, yaw)
                    except ValueError:
                        print("Invalid input. Please provide numeric values.")

            elif choice == "2":
                return (0.0, 0.0, 0.0)

            elif choice == "3":
                raise SystemExit("Exiting optimization: no start pose selected.")

            else:
                print("Invalid choice. Please enter 1, 2, or 3.")


    def _resolve_robot_start_pose(
        self,
        metadata: dict,
        prompt_missing: bool,
    ) -> Tuple[float, float, float]:
        start_pose = self._coerce_pose_tuple(metadata.get("robot_start_pose"))
        if start_pose is not None:
            return start_pose

        if prompt_missing:
            return self._prompt_start_pose_fallback()

        return (0.0, 0.0, 0.0)


    def _load_step_ids(self, n_steps: Optional[int] = None):
        step_ids = set()

        with open(self.steps_path, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                if n_steps is not None and len(step_ids) >= n_steps:
                    break

                step_ids.add(int(row["step_id"]))

        return step_ids


    def _load_scans(self, step_ids=None):
        scan_dict = {}

        filter_enabled = step_ids is not None

        with open(self.scans_path, "r") as f:
            for line in f:
                entry = json.loads(line)

                step_id = entry["step_id"]

                if filter_enabled and step_id not in step_ids:
                    continue

                scan_dict[step_id] = RawScan(
                    angle_min=entry["angle_min"],
                    angle_increment=entry["angle_increment"],
                    range_min=entry["range_min"],
                    range_max=entry["range_max"],
                    ranges=np.array(entry["ranges"], dtype=np.float32),
                )

        return scan_dict


    def _load_steps(self, scan_dict, n_steps: Optional[int] = None):
        steps = []

        with open(self.steps_path, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                if n_steps is not None and len(steps) >= n_steps:
                    break

                step_id = int(row["step_id"])

                step = PlaybackStep(
                    step_id=step_id,
                    t=float(row["t"]),
                    t_ros=float(row["t_ros"]),
                    dl=float(row["dl"]),
                    dr=float(row["dr"]),
                    true_pose=(
                        float(row["x"]),
                        float(row["y"]),
                        float(row["theta"]),
                    ),
                    scan=scan_dict[step_id],
                )

                steps.append(step)

        return steps