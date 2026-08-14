
from dataclasses import dataclass
from typing import List, Optional, Tuple

import os
import csv
import json
import numpy as np


@dataclass
class RawScan:
    '''
    Dataclass representing a raw laser scan, including angle parameters and range measurements.
    '''
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float
    ranges: np.ndarray


@dataclass
class PlaybackStep:
    '''
    Dataclass representing a single step in the playback data, including time, odometry, true pose, and associated scan.
    '''
    step_id: int
    t: float
    t_ros: float
    dl: float
    dr: float
    true_pose: Tuple[float, float, float]
    scan: RawScan


@dataclass
class PlaybackData:
    '''
    Dataclass representing the entire playback data, including metadata and a list of steps.
    '''
    metadata: dict
    steps: List[PlaybackStep]


class PlaybackLoader:
    '''
    Class responsible for loading playback data from files. It reads metadata, scans, and steps from specified files
    and returns a structured PlaybackData object.
    '''
    def load(
        self,
        file_suffix,
        filedir,
        n_steps: Optional[int] = None,
        ensure_start_pose: bool = False,
        prompt_for_missing_start_pose: bool = False,
    ) -> PlaybackData:
        '''
        Loads the playback data from the given file suffix and directory, if exists. Returns a PlaybackData object
        containing metadata and steps.

        Parameters
        ----------
        file_suffix : str
            The suffix of the playback files to load (e.g., "playback_2023_08_15").
        filedir : str
            The directory where the playback files are located.
        n_steps : Optional[int], optional
            The number of steps to load. If None, all steps are loaded. Must be >= 0 if provided. Default is None.
        ensure_start_pose : bool, optional
            If True, ensures that the robot's start pose is present in the metadata. If missing, it will be resolved
            based on the prompt_for_missing_start_pose flag. Default is False.
        prompt_for_missing_start_pose : bool, optional
            If True and ensure_start_pose is True, prompts the user to define a start pose if it is missing in the metadata.
            If False, defaults to (0.0, 0.0, 0.0). Default is False.

        Returns
        -------
        PlaybackData
            An object containing the loaded metadata and steps.
        '''
        # Define paths
        self.meta_path = os.path.join(filedir, f"{file_suffix}_meta.json")
        self.scans_path = os.path.join(filedir, f"{file_suffix}_scans.jsonl")
        self.steps_path = os.path.join(filedir, f"{file_suffix}_steps.csv")

        if n_steps is not None and n_steps < 0:
            raise ValueError(f"n_steps must be >= 0 or None, got {n_steps}")

        # Load metadata from playback files
        metadata = self._load_metadata()

        # Extract start pose from metadata and resolve if missing
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
        '''
        Coerce a value to a tuple of three floats representing a pose (x, y, yaw).

        Returns
        -------
        Optional[Tuple[float, float, float]]
            The coerced pose tuple if valid, otherwise None.
        '''
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return None

        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            return None


    @staticmethod
    def _prompt_start_pose_fallback() -> Tuple[float, float, float]:
        """
        Prompt the user to define a start pose or use a fallback.

        Returns
        -------
        Tuple[float, float, float]
            The chosen start pose as a tuple (x, y, yaw).
        """
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
        '''
        Resolve the robot's start pose from metadata. If missing, prompt the user for action or fallback to zero pose.
        '''
        # Get start pose
        start_pose = self._coerce_pose_tuple(metadata.get("robot_start_pose"))

        # Return if exists
        if start_pose is not None:
            return start_pose

        # If missing, prompt user for action or fallback to zero pose
        if prompt_missing:
            return self._prompt_start_pose_fallback()

        return (0.0, 0.0, 0.0)


    def _load_step_ids(self, n_steps: Optional[int] = None):
        '''
        Load step IDs from the steps CSV file. If n_steps is specified, limit the number of step IDs returned.

        Parameters
        ----------
        n_steps : Optional[int], optional
            The number of step IDs to load. If None, all step IDs are loaded. Must

        Returns
        -------
        Set[int]
            A set of step IDs.
        '''
        step_ids = set()

        with open(self.steps_path, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                if n_steps is not None and len(step_ids) >= n_steps:
                    break

                step_ids.add(int(row["step_id"]))

        return step_ids


    def _load_scans(self, step_ids=None):
        '''
        Load the laser scans from the scans JSONL file. If step_ids is provided, only load scans for those step IDs.

        Parameters
        ----------
        step_ids : Optional[Set[int]], optional
            A set of step IDs to filter the scans. If None, all scans are loaded.

        Returns
        -------
        Dict[int, RawScan]
            A dictionary mapping step IDs to RawScan objects.
        '''
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
        '''
        Loads the step data from the steps CSV file and associates each step with its corresponding scan from scan_dict.

        Parameters
        ----------
        scan_dict : Dict[int, RawScan]
            A dictionary mapping step IDs to RawScan objects.
        n_steps : Optional[int], optional
            The number of steps to load. If None, all steps are loaded. Must be >= 0 if provided. Default is None.
        
        Returns
        -------
        List[PlaybackStep]
            A list of PlaybackStep objects containing the loaded step data.
        '''
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