#!/usr/bin/env python3
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==== FILE PATHS ====
INPUT_FILE = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/scan_match/python_playback/1776425398_python_playback_steps.csv"
OUTPUT_FILE = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/scan_match/python_playback/1776425398_with_speed.csv"

# ==== COLUMN NAMES ====
COL_TIME = "t"
COL_X = "true_pose_x"
COL_Y = "true_pose_y"
COL_THETA = "true_pose_yaw"
COL_TRANS_SPEED = "v"
COL_ROT_SPEED = "omega"


def normalize_angle(angle):
    """
    Normalize angle to [-pi, pi]
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def compute_speeds(df):
    # Previous values
    x_prev = df[COL_X].shift(1)
    y_prev = df[COL_Y].shift(1)
    theta_prev = df[COL_THETA].shift(1)
    t_prev = df[COL_TIME].shift(1)

    # Differences
    dx = df[COL_X] - x_prev
    dy = df[COL_Y] - y_prev
    dtheta = df[COL_THETA] - theta_prev

    # Normalize angle difference
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    d_theta_deg = np.degrees(dtheta)

    dt = df[COL_TIME] - t_prev

    # Avoid division by zero
    dt_safe = np.where(dt == 0, 1e-6, dt)

    # Translational speed
    dist = np.sqrt(dx**2 + dy**2)
    v = dist / dt_safe

    # Rotational speed
    omega = d_theta_deg / dt_safe

    # First row fix
    v.iloc[0] = 0.0
    omega.iloc[0] = 0.0

    return v, omega


def analyze_speed(df: pd.DataFrame):
    # Get min, max, mean speed
    min_v = df['v'].min()
    max_v = df['v'].max()
    mean_v = df['v'].mean()

    min_omega = df['omega'].min()
    max_omega = df['omega'].max()
    mean_omega = df['omega'].mean()


def analyze_speed_v2(df: pd.DataFrame, headers: List[str]):
    summary = df[headers].describe()
    print("\n\nSummary of speed Analysis:\n")
    print(summary)

    # Plot speed over time
    plt.plot(df["t"], df["v"], label="v [m/s]", color="b")
    plt.plot(df["t"], df['omega'], label="omega[deg/s]", color="g")
    plt.xlabel("t")
    plt.ylabel("speed")

    plt.legend()
    plt.grid()
    plt.show()


def analyze_speed_v3(df: pd.DataFrame, headers: List[str]):
    summary = df[headers].describe()
    print("\n\nSummary of speed Analysis:\n")
    print(summary)

    fig, ax1 = plt.subplots()

    # ---- Left axis (v) ----
    ax1.plot(df["t"], df["v"], label="v [m/s]", color="b")
    ax1.set_xlabel("t")
    ax1.set_ylabel("v [m/s]", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    # ---- Right axis (omega) ----
    ax2 = ax1.twinx()
    ax2.plot(df["t"], df["omega"], label="omega [deg/s]", color="g")
    ax2.set_ylabel("omega [deg/s]", color="g")
    ax2.tick_params(axis="y", labelcolor="g")

    # ---- Legend (combine both) ----
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2)

    plt.grid()
    plt.title("Translational vs Rotational Speed")
    plt.show()



def main():
    df = pd.read_csv(INPUT_FILE)

    # Compute speeds
    v, omega = compute_speeds(df)

    # Add to dataframe
    headers = ["v", "omega"]
    df["v"] = v
    df["omega"] = omega

    # Analyze data
    analyze_speed_v3(df, headers)
    

    # Save
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Saved file: {OUTPUT_FILE}")

    # Debug preview
    print("\nPreview:")
    print(df[[COL_TIME, COL_X, COL_Y, COL_THETA, "v", "omega"]].head(10))

    print("\nStats:")
    print("v max:", df["v"].max())
    print("omega max:", df["omega"].max())


if __name__ == "__main__":
    main()