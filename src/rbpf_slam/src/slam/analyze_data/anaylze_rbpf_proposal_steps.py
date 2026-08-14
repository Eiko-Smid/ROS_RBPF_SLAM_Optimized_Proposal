#!/usr/bin/env python3
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==== FILE PATHS ====
PAYBACK_STEPS_FILE = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/python_playback/1777891056_steps.csv"
RBPF_STEPS_FILE = '/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optimization_results/1777891056_optm_23_1_steps.csv'
OUTPUT_FILE = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optimization_results/1777891056_cafe_proposal_step_analysis.csv"

RANK = 1
N_XJ = 27.0

# ==== COLUMN NAMES ====
# COL_TIME = "t"
# COL_X = "true_pose_x"
# COL_Y = "true_pose_y"
# COL_THETA = "true_pose_yaw"
# COL_TRANS_SPEED = "v"
# COL_ROT_SPEED = "omega"

COL_TIME = "t"              # Time in seconds. TODO: Transfer into ms here!
COL_X = "x"
COL_Y = "y"
COL_THETA = "theta"
COL_TRANS_SPEED = "v"
COL_ROT_SPEED = "omega"

COL_SPEED = ["step_id", "v", "omega"]
COL_STEPS = ["scan_match_fallback_failed", "neff", "trans_error", "rot_error"]
COL_RESUTLS = [
    "step_id",
    "scan_match_fallback_failed", 
    "neff", 
    "trans_error", "rot_error", 
    "v", "omega"
]


# COL_SPEED = ["step_id", "v", "omega"]
COL_STEPS_PROPOSAL = [
    "step_id",
    "scan_match_failed_count", 
    "trans_error",
    "rot_error_deg",
    "trans_err_sm_true", 
    "rot_err_sm_true_deg",
    "trans_err_min_xj_true",
    "xj_eff",
]
COL_RESUTLS_PROPOSAL = [
    "step_id",
    "scan_match_failed_count", 
    "trans_err_sm_true", 
    "rot_err_sm_true_deg",
    "xj_eff",
]


N_PARTICLES = 40.0
NEFF_THRESHOLD = N_PARTICLES / 2.0



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

    # Normalize angle difference and transform to degrees
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    d_theta_deg = np.degrees(dtheta)

    # Compute dt
    dt = df[COL_TIME] - t_prev

    # Avoid division by zero
    dt_safe = np.where(dt == 0, 1e-6, dt)

    # Translational speed
    dist = np.sqrt(dx**2 + dy**2)
    v = dist / dt_safe

    # Rotational speed
    omega = d_theta_deg / dt_safe

    # Add zeros for speed to first row
    v.iloc[0] = 0.0
    omega.iloc[0] = 0.0

    return v, omega


def compute_speeds_copy(df):
    # Previous values
    x_prev = df[COL_X].shift(1)
    y_prev = df[COL_Y].shift(1)
    theta_prev = df[COL_THETA].shift(1)
    t_prev = df[COL_TIME].shift(1)

    # Differences
    dx = df[COL_X] - x_prev
    dy = df[COL_Y] - y_prev
    dtheta = df[COL_THETA] - theta_prev

    # Normalize angle difference and transform to degrees
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    d_theta_deg = np.degrees(dtheta)

    # Compute dt
    dt = df[COL_TIME] - t_prev

    # Avoid division by zero
    dt_safe = np.where(dt == 0, 1e-6, dt)

    # Translational speed
    dist = np.sqrt(dx**2 + dy**2)
    v = dist / dt_safe

    # Rotational speed
    omega = d_theta_deg / dt_safe

    # Add zeros for speed to first row
    v.iloc[0] = 0.0
    omega.iloc[0] = 0.0

    return v, omega


def analyze_speed(df: pd.DataFrame, headers: List[str]):
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


def combine_data_to_result(
        df_speed: pd.DataFrame,
        df_steps: pd.DataFrame,
        col_speed: List[str],
        col_steps: List[str],
        col_results: List[str]
) -> pd.DataFrame:  
    # Extract relevant columns from dfs
    df_speed_subset = df_speed[col_speed]
    df_steps_subset = df_steps[col_steps]

    # Merge together
    df_comb = pd.concat([df_speed_subset, df_steps_subset], axis=1)
    
    # Store relevant col in resutls df
    df_results = df_comb[col_results]

    return df_results


def compute_useful_metrics(df_result: pd.DataFrame, resample_threshold: float):
    # Compute resampling indicator based on neff threshold
    df_result["resampled"] = (df_result["neff"] < resample_threshold).astype(int)
    return df_result


def plot_results(df_result: pd.DataFrame):
    # Define figure and axis
    fig, ax1 = plt.subplots()

    # ---- Left axis (v) ----
    ax1.plot(df_result["step_id"], df_result["v"], label="v [m/s]", color="b")
    ax1.plot(df_result["step_id"], df_result["neff"], label="neff", color="g")
    ax1.plot(df_result["step_id"], df_result["trans_error"], label="trans_error [m]", color="r")
    ax1.plot(df_result["step_id"], df_result["rot_error"], label="rot_error [deg]", color="c")
    ax1.set_xlabel("step_id")
    ax1.set_ylabel("v [m/s]", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    # ---- Right axis (omega) ----
    ax2 = ax1.twinx()
    ax2.plot(df_result["step_id"], df_result["omega"], label="omega [deg/s]", color="g")
    ax2.set_ylabel("omega [deg/s]", color="g")
    ax2.tick_params(axis="y", labelcolor="g")

    # ---- Legend (combine both) ----
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2)

    plt.grid()
    plt.title("RBPF Step Analysis")
    plt.show()


def plot_results_v2(df):
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

    # --- 1. Motion ---
    axs[0].plot(df["step_id"], df["v"], label="v [m/s]", color="b")
    axs[0].set_ylabel("v [m/s]", color="b")

    ax1b = axs[0].twinx()
    ax1b.plot(df["step_id"], df["omega"], label="omega [deg/s]", color="g")
    ax1b.set_ylabel("omega [deg/s]", color="g")

    axs[0].set_title("Motion")

    # --- 2. Errors ---
    axs[1].plot(df["step_id"], df["trans_error"], label="trans_error [m]", color="r")
    axs[1].plot(df["step_id"], df["rot_error"], label="rot_error [deg]", color="c")
    axs[1].set_ylabel("Error")
    axs[1].legend()
    axs[1].set_title("Estimation Error")

    # --- 3. Filter state ---
    axs[2].step(df["step_id"], df["resampled"], where="post", label="resampling", color="black")
    axs[2].set_ylim(-0.1, 1.1)
    # axs[2].axhline(neff_threshold, linestyle="--", color="gray", label="resample threshold")
    axs[2].set_ylabel("neff")
    axs[2].set_xlabel("step")
    axs[2].legend()
    axs[2].set_title("Filter State")

    plt.tight_layout()
    plt.show()


def analyze_rbpf_proposal_run(rank: int):
    # Load data
    df_playback = pd.read_csv(PAYBACK_STEPS_FILE)
    df_steps = pd.read_csv(RBPF_STEPS_FILE)

    # Filter step data by rank
    df_steps = df_steps[df_steps["rank"] == rank]
    
    # Convert time s -> ms
    df_playback[COL_TIME] = df_playback[COL_TIME] / 1000.0

    # Filter NaN values 
    # df_steps = df_steps.dropna(subset=COL_STEPS_PROPOSAL)

    # Convert boolean indicator to int
    df_steps["scan_match_failed"] = df_steps["scan_match_failed"].astype(int)

    # Normalize xj-eff to [0, 1]
    df_steps["xj_eff"] = df_steps["xj_eff"] / N_XJ

    # Plot data
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # set x ticks
    step_ticks = np.arange(0, df_steps["step_id"].max() + 1, 25)

    # --- 1. Errors and xj_eff ---
    axs[0].plot(df_steps["step_id"], df_steps["trans_err_sm_true"], label="trans_err_sm_true [m]", color="r")
    axs[0].plot(df_steps["step_id"], df_steps["rot_err_sm_true_deg"], label="rot_err_sm_true [deg]", color="c")
    axs[0].plot(df_steps["step_id"], df_steps["xj_eff"], label="xj_eff", color="m")
    axs[0].set_ylabel("Error")
    axs[1].set_xlabel("step")
    axs[0].set_xticks(step_ticks)
    axs[0].legend(loc="upper right")
    axs[0].grid()
    axs[0].set_title("Proposal Estimation Error")

    # Plot scan match fallback failure
    axs[1].step(df_steps["step_id"], df_steps["scan_match_failed"], where="post", label="scan match failed", color="black")
    axs[1].set_ylim(-0.1, 1.1)
    axs[1].set_ylabel("scan match failed")
    axs[1].set_xlabel("step")
    axs[1].set_xticks(step_ticks)
    axs[1].legend(loc="upper right")
    axs[1].set_title("Scan Match Failure")

    # plt.tight_layout()
    plt.grid()
    plt.show()
    

def analyze_rbpf_run(rank: int):
    # Load data
    df_playback = pd.read_csv(PAYBACK_STEPS_FILE)
    df_steps = pd.read_csv(RBPF_STEPS_FILE)

    # Filter step data by rank
    df_steps = df_steps[df_steps["rank"] == rank]
    
    # CConvert time s -> ms
    df_playback[COL_TIME] = df_playback[COL_TIME] / 1000.0

    print(df_playback.head())

    # Compute speed columns
    v, omega = compute_speeds(df_playback)

    # Define speed df
    df_speed = df_playback.copy()
    df_speed["v"] = v
    df_speed["omega"] = omega

    # Merge playback with steps
    df_results = combine_data_to_result(
        df_speed=df_speed,
        df_steps=df_steps,
        col_steps=COL_STEPS,
        col_speed=COL_SPEED,
        col_results=COL_RESUTLS
    )

    # Compute useful metrics (e.g. resampling indicator)
    df_results = compute_useful_metrics(df_results, resample_threshold=NEFF_THRESHOLD)

    # Plot result data
    plot_results_v2(df_results)

    # Store data to csv
    df_results.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved file: {OUTPUT_FILE}")




def analyze_speed_data():
    # Load step data
    df = pd.read_csv(PAYBACK_STEPS_FILE)
    print(df.head())
    
    # Compute speeds
    v, omega = compute_speeds(df)

    # Add to dataframe
    headers = ["v", "omega"]
    df["v"] = v
    df["omega"] = omega

    # Analyze data
    analyze_speed(df, headers)
    
    # Save
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Saved file: {OUTPUT_FILE}")

    # Debug preview
    print("\nPreview:")
    print(df[[COL_TIME, COL_X, COL_Y, COL_THETA, "v", "omega"]].head(10))

    print("\nStats:")
    print("v max:", df["v"].max())
    print("omega max:", df["omega"].max())


def main():
    # analyze_rbpf_run(rank=RANK)
    analyze_rbpf_proposal_run(rank=RANK)


if __name__ == "__main__":
    main()