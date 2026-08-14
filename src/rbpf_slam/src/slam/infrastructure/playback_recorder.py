import os
import csv
import json
from dataclasses import asdict
import time


class PlaybackRecorder:
    def __init__(self, output_dir: str, metadata: dict):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        timestamp = str(int(time.time()))
        filename_steps = f"{timestamp}_steps.csv"
        filename_scans = f"{timestamp}_scans.jsonl"
        filename_metadata = f"{timestamp}_meta.json"
        
        self.steps_csv_path = os.path.join(output_dir, filename_steps)
        self.scans_json_path = os.path.join(output_dir, filename_scans)
        self.metadata_path = os.path.join(output_dir, filename_metadata)

        self.step_id = 0

        # --- Init CSV ---
        self._init_csv()

        # --- Write metadata once ---
        self._write_metadata(metadata)


    def _init_csv(self):
        with open(self.steps_csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step_id",
                "t",
                "t_ros",
                "dl",
                "dr",
                "x",
                "y",
                "theta",
            ])


    def _write_metadata(self, metadata: dict):
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)


    def record_step(
        self,
        t: float,
        t_ros: float,
        dl: float,
        dr: float,
        true_pose,
        laser_scan
    ):

        # --- Write step CSV ---
        with open(self.steps_csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.step_id,
                t,
                t_ros,
                dl,
                dr,
                true_pose[0],
                true_pose[1],
                true_pose[2],
            ])

        # --- Write scan JSONL ---
        scan_entry = {
            "step_id": self.step_id,
            "angle_min": laser_scan.angle_min,
            "angle_increment": laser_scan.angle_increment,
            "range_min": laser_scan.range_min,
            "range_max": laser_scan.range_max,
            "ranges": list(laser_scan.ranges),
        }

        with open(self.scans_json_path, mode="a") as f:
            f.write(json.dumps(scan_entry) + "\n")

        self.step_id += 1