#!/usr/bin/env python3

import itertools
import json
import numpy as np
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterator, List, Union

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

from ..optimize_rbpf.playback_defs import ExperimentParams
from .evaluator_scanmatching import ScanMatchingEvaluator
from .playback_runner_scanmatching import PlaybackRunnerScanMatching
from .scorer_scanmatching import ScanMatchingScorer
from .optimizer_scanmatching import ScanMatchingOptimizer
from .result_writer_scanmatching import ResultWriterScanMatching

'''
1. Test final pipeline
    1.1 Test of final tuning pipeline for scan-matching-only mode on 100 steps 

    1.2 Test after correction and metric changes

        - First test of full run after correction and metric changes. 



'''

SCAN_MATCHING_RESULT_PATH = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/scan_matching/optimization_results/sm_1_2_summary.csv"
SCAN_MATCHING_STEP_TRACE_PATH = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/scan_matching/optimization_results/sm_1_2_trace_steps.csv"
PARAMETER_OVERVIEW_PATH = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/scan_matching/optimization_results/sm_1_2_params.json"

# # SCAN_MATCHING_RESULT_PATH = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/scan_matching/optimization_results/sm_copilot_test_full_step_summary.csv"
# # SCAN_MATCHING_STEP_TRACE_PATH = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/scan_matching/optimization_results/sm_copilot_test_full_steps_trace.csv"
# # PARAMETER_OVERVIEW_PATH = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/scan_matching/optimization_results/sm_copilot_test_full_steps_params.json"

CSV_FLOAT_DECIMALS = 5
OVERRIDE_EXISTING_RESULTS = False
N_PLAYBACK_STEPS = None
N_OPTIMIZATION_REPEATS = 1
# SEED_LIST = [22, 23, 24, 56]
SEED_LIST = [22]

PLAYBACK_DIR = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/scan_matching/python_playback/"
PLAYBACK_SUFFIX = "1777891056"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _compute_wheel_separation() -> float:
    h_chassis = 0.15
    dist_chassis_to_ground = h_chassis / 5
    r_wheel = h_chassis / 2 + dist_chassis_to_ground
    w_wheel = 0.3 * r_wheel
    r_chassis = 0.25
    return 2 * r_chassis + w_wheel


def _grid_axes() -> Dict[str, List[Union[float, int]]]:
    return {
        "every_nth_beam_filter": [4],
        "every_nth_beam_map": [2],
        "max_correspondence_distance": [0.6],
        "max_translation_jump": [0.8],
        "max_rotation_jump_deg": [120.0],
    }


def write_parameter_overview(path: str, n_repeats: int, override: bool = False) -> None:
    file_exists = ResultWriterScanMatching.create_path_and_check_if_file_exists(path=path)

    if file_exists and not override:
        print("\nParameter overview has not been saved because file already exists and override is set to False!")
        return

    axes = _grid_axes()
    example_params = next(generate_param_grid(n_repeats=1), None)

    payload = {
        "playback_dir": PLAYBACK_DIR,
        "playback_suffix": PLAYBACK_SUFFIX,
        "n_playback_steps": N_PLAYBACK_STEPS,
        "n_optimization_repeats": n_repeats,
        "seed_list": SEED_LIST,
        "grid_axes": axes,
        "example_experiment_params": _to_jsonable(example_params) if example_params is not None else None,
    }

    with open(path, "w") as f:
        json.dump(_to_jsonable(payload), f, indent=2)

    print(f"\nParameter overview has been saved to:\n{path}")


def generate_param_grid(n_repeats: int = 1) -> Iterator[ExperimentParams]:
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    axes = _grid_axes()
    wheel_separation = _compute_wheel_separation()

    for repeat_idx in range(1, n_repeats + 1):
        for every_nth_filter, every_nth_map, max_corr_dist, max_jump_trans, max_jump_rot_deg in itertools.product(
            axes["every_nth_beam_filter"],
            axes["every_nth_beam_map"],
            axes["max_correspondence_distance"],
            axes["max_translation_jump"],
            axes["max_rotation_jump_deg"],
        ):
            yield ExperimentParams(
                occupancy_params=OccupancyParams(
                    # All cells are initalized with this probability when map is initialized. 
                    prior_probability=0.5, 
                    # Min distance of the robot to the border of the map. If robot is closer than this to the border, the map will be extended.                   
                    min_distance_to_border=10.0,
                    # Cell is increased by the log Odds of this value when beam ends in this cell
                    increasing_probability=0.7,
                    # All cells a beam passed will decreased by the log Odds of this value  
                    decreasing_probability=0.30,
                    # Max and Min possible log odds value a cell can have. 
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
                    n_particles=1,
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
                every_nth_scan_map=every_nth_map,
                proposal_sigma_xy=1.0,
                proposal_sigma_theta=1.0,
                proposal_n_samples=1,
                tag=(
                    f"every_nth_filter{every_nth_filter}_every_nth_map{every_nth_map}_"
                    f"mcd{max_corr_dist}_mjt{max_jump_trans}_mjrd{max_jump_rot_deg}_rep{repeat_idx}"
                ),
            )


def build_optimizer() -> ScanMatchingOptimizer:
    runner = PlaybackRunnerScanMatching(
        factory=RBPFFactory(),
        evaluator=ScanMatchingEvaluator(),
    )

    return ScanMatchingOptimizer(
        runner=runner,
        scorer=ScanMatchingScorer(),
    )


def main() -> None:
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

    # Store compact parameter overview (grid axes + one representative ExperimentParams)
    write_parameter_overview(
        path=PARAMETER_OVERVIEW_PATH,
        n_repeats=N_OPTIMIZATION_REPEATS,
        override=OVERRIDE_EXISTING_RESULTS,
    )

    ranked_runs = optimizer.optimize(
        playback_data=playback_data,
        param_grid=generate_param_grid(n_repeats=N_OPTIMIZATION_REPEATS),
        seeds=SEED_LIST,
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
