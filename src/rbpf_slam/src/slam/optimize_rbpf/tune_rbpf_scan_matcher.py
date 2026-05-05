#!/usr/bin/env python3

import itertools
import numpy as np

from ..infrastructure.playback_loader import PlaybackLoader
from ..infrastructure.playback_converter import PlaybackConverter

from ..rbpf.rbpf import RBPFFactory, ParticleParams, MotionModelParams, MeasurementModelParams
from ..rbpf.scan_match_factory import (
    OccupancyParams,
    SensorParams,
    MapParameter,
    ICPParams,
    RobotParams,
    ScanMatcherParams,
)

from .playback_defs import ExperimentParams
from .evaluator import RBPFEvaluator
from .playback_runner_scanmatching import PlaybackRunnerScanMatching
from .scorer_scanmatching import ScanMatchingScorer
from .optimizer import ScanMatcherOptimizer
from .result_writer_scanmatching import ResultWriterScanMatching


SCAN_MATCHING_RESULT_PATH = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optimization_results/scan_matching_only_summary.csv"
SCAN_MATCHING_STEP_TRACE_PATH = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optimization_results/scan_matching_only_steps.csv"

CSV_FLOAT_DECIMALS = 4
OVERRIDE_EXISTING_RESULTS = False
N_PLAYBACK_STEPS = 20
N_OPTIMIZATION_REPEATS = 1
BASE_SEED = 22
RESEED_EACH_RUN = True

PLAYBACK_DIR = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/python_playback/"
PLAYBACK_SUFFIX = "1777891056"


def _compute_wheel_separation() -> float:
    h_chassis = 0.15
    dist_chassis_to_ground = h_chassis / 5
    r_wheel = h_chassis / 2 + dist_chassis_to_ground
    w_wheel = 0.3 * r_wheel
    r_chassis = 0.25
    return 2 * r_chassis + w_wheel


def _grid_axes() -> dict:
    return {
        "every_nth_beam_filter": [4],
        "n_particles": [1],
        "max_correspondence_distance": [0.6],
        "max_translation_jump": [0.8],
        "max_rotation_jump_deg": [120.0],
    }


def generate_param_grid(n_repeats: int = 1):
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    axes = _grid_axes()
    wheel_separation = _compute_wheel_separation()

    for repeat_idx in range(1, n_repeats + 1):
        for every_nth_filter, n_part, max_corr_dist, max_jump_trans, max_jump_rot_deg in itertools.product(
            axes["every_nth_beam_filter"],
            axes["n_particles"],
            axes["max_correspondence_distance"],
            axes["max_translation_jump"],
            axes["max_rotation_jump_deg"],
        ):
            yield ExperimentParams(
                occupancy_params=OccupancyParams(
                    prior_probability=0.5,
                    min_distance_to_border=10.0,
                    increasing_probability=0.7,
                    decreasing_probability=0.30,
                    min_log_odds=-5.0,
                    max_log_odds=5.0,
                ),
                sensor_params=SensorParams(
                    min_sensor_range=0.1,
                    max_sensor_range=10.0,
                ),
                map_param=MapParameter(
                    map_width=10.0,
                    map_height=10.0,
                    grid_resolution_m=0.1,
                ),
                icp_params=ICPParams(
                    max_n_points=400,
                    max_correspondence_distance=max_corr_dist,
                    neighbors_pca=10,
                    max_iterations=5,
                    epsilon_rel=1e-3,
                    no_improvement_limit=3,
                    min_error=5e-4,
                    min_dtrans=1e-3,
                    min_drot=1e-2,
                    min_points=20,
                    min_corresp=15,
                    min_hessian_rank=3,
                    max_hessian_condition=1e8,
                    max_translation_jump=max_jump_trans,
                    max_rotation_jump=np.deg2rad(max_jump_rot_deg),
                    max_acceptable_mean_error=0.15,
                ),
                robot_params=RobotParams(
                    wheel_separation=wheel_separation,
                ),
                scan_matcher_params=ScanMatcherParams(
                    occ_thres=1.2,
                    delta_r=0.6,
                ),
                particle_params=ParticleParams(
                    n_particles=n_part,
                    start_pose=(0.0, 0.0, 0.0),
                ),
                # Unused in scan-matching-only mode but required by ExperimentParams.
                motion_model_params=MotionModelParams(
                    sigma_x=0.2,
                    sigma_y=0.2,
                    sigma_theta=0.15,
                    wheel_separation=wheel_separation,
                    ctrl_motion_fac=0.1,
                    ctrl_turn_fac=0.15,
                ),
                # Unused in scan-matching-only mode but required by ExperimentParams.
                measurement_model_params=MeasurementModelParams(
                    sigma_measurement=0.25,
                ),
                every_nth_scan_filter=every_nth_filter,
                every_nth_scan_map=2,
                proposal_sigma_xy=1.0,
                proposal_sigma_theta=1.0,
                proposal_n_samples=1,
                tag=(
                    f"sm_only_nthf{every_nth_filter}_npart{n_part}_"
                    f"mcd{max_corr_dist}_mjt{max_jump_trans}_mjrd{max_jump_rot_deg}_rep{repeat_idx}"
                ),
            )


def build_optimizer():
    runner = PlaybackRunnerScanMatching(
        factory=RBPFFactory(),
        evaluator=RBPFEvaluator(),
    )

    return ScanMatcherOptimizer(
        runner=runner,
        scorer=ScanMatchingScorer(),
    )


def main():
    if BASE_SEED is not None:
        np.random.seed(BASE_SEED)

    playback_loader = PlaybackLoader()
    raw_playback_data = playback_loader.load(
        file_suffix=PLAYBACK_SUFFIX,
        filedir=PLAYBACK_DIR,
        n_steps=N_PLAYBACK_STEPS,
    )

    playback_conv = PlaybackConverter()
    playback_data = playback_conv.convert(raw_playback_data)

    optimizer = build_optimizer()
    writer = ResultWriterScanMatching()

    ranked_runs = optimizer.optimize(
        playback_data=playback_data,
        param_grid=generate_param_grid(n_repeats=N_OPTIMIZATION_REPEATS),
        base_seed=BASE_SEED,
        reseed_each_run=RESEED_EACH_RUN,
    )

    writer.write_ranked_runs_csv(
        path=SCAN_MATCHING_RESULT_PATH,
        ranked_runs=ranked_runs,
        override=OVERRIDE_EXISTING_RESULTS,
        float_decimals=CSV_FLOAT_DECIMALS,
    )

    writer.write_ranked_step_traces_csv(
        output_path=SCAN_MATCHING_STEP_TRACE_PATH,
        ranked_runs=ranked_runs,
        override=OVERRIDE_EXISTING_RESULTS,
        float_decimals=CSV_FLOAT_DECIMALS,
    )

    print("Scan-matching-only tuning run finished.")


if __name__ == "__main__":
    main()
