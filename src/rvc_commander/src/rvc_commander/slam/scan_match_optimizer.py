#!/usr/bin/env python3

from playback_loader import load_playback_dataset

BASE = "/home/smide/work/ros_workspaces/ros_ws/src/rvc_commander/data/scan_match/python_playback/1776252417_python_playback"


def test():
    map_data, steps = load_playback_dataset(BASE)

    # Print dataset info
    print(f"\nMap shape: {map_data.log_odds_map.shape}")
    print(f"Number of steps: {len(steps)}")

    print(f"First step true pose: {steps[0].true_pose}")
    print(f"Number of scans in first step: {len(steps[0].scan)}")


def main():
    test()


if __name__ == "__main__":
    main()  