#!/usr/bin/env python3

# import debugpy
# debugpy.listen(("localhost", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

import itertools
import json
import numpy as np
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterator, List, Tuple, Union

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


2. Optimize further
    - We added new metrics, increased the grid parameters and added new parameters to the grid. 
    - Also we computed search window for surface detection in map extractor 

    2.1 Over cafe map

    2.2 over turtlebot map

    
3. After intial shift fix and after adding initalization pahse for algorithm

    3.1 Cafe map (1779375646)
    
        3.1.1 First run on cafe map with zero stddev in scan measurements
            Results:
                - Solid results

        3.1.2 Run on cafe map with added noise in scan ranges (accidently without measurement seed)

        
        3.1.3 Run on cafe map with added noise in scan ranges with measurement seed

            - Quiet good results but a little bit more worse then the corresponding turtle bot 3 results.
            - We did smaller turns here but scan matcher weakness is not turn as it seems its more translation.
            - Also this map might not have that much featueres than the turtle bot map has. 
            
        3.1.4 Run on cafe map with on small grid with different seeds -> find stable params

        
    3.2 turtle bot 3 map (1779363559)

        3.2.1 First run on turtle bot map with zero stddev in scan measurements
            Results:
                - Solid results

        3.2.2 Run on turtle bot map with added noise in scan ranges (accidently without measurement seed)

        
        3.2.3 Run on turtle bot map with added noise in scan ranges with measurement seed
        
        3.2.4 Run on turtle bot map with on small grid with different seeds -> find stable params
'''


SCAN_MATCHING_RESULT_PATH = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/scan_matching/optimization_results/sm_1779363559_3_2_4_summary.csv"
SCAN_MATCHING_STEP_TRACE_PATH = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/scan_matching/optimization_results/sm_1779363559_3_2_4_trace_steps.csv"
PARAMETER_OVERVIEW_PATH = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/scan_matching/optimization_results/sm_1779363559_3_2_4_params.json"

# SCAN_MATCHING_RESULT_PATH = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/scan_matching/optimization_results/sm_test_7_summary.csv"
# SCAN_MATCHING_STEP_TRACE_PATH = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/scan_matching/optimization_results/sm_test_7_trace_steps.csv"
# PARAMETER_OVERVIEW_PATH = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/scan_matching/optimization_results/sm_test_7_params.json"

CSV_FLOAT_DECIMALS = 5
OVERRIDE_EXISTING_RESULTS = False
N_PLAYBACK_STEPS = None
N_OPTIMIZATION_REPEATS = 1
SEED_LIST = [22, 23, 56]
# SEED_LIST = [22]

# Controls ONLY measurement-noise seeding behavior in optimizer:
# - True:  use values from SEED_LIST for deterministic per-seed measurement noise.
# - False: do not seed measurement noise (fresh random noise every run).
USE_SEED_LIST_FOR_MEASUREMENT_NOISE = True

# Define sttdev [m] to add noise to the playback measurement. This is only possible if the playback data doesnt include noise in the measurements.
MEASUREMENT_STDDEV = 0.03
MIN_SENSOR_RANGE = 0.1
MAX_SENSOR_RANGE = 10.0 

PLAYBACK_DIR = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/scan_matching/python_playback/"
PLAYBACK_SUFFIX = "1779363559"        # turtlebot 3 map
# PLAYBACK_SUFFIX = "1779375646"          # Cafe map    


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
        # Playback sampling
        "every_nth_beam_filter": [4],
        "every_nth_beam_map": [2],

        # OccupancyParams (OGM)
        "increasing_probability": [0.7, 0.85],
        "decreasing_probability": [0.30, 0.15],
        "min_log_odds": [-5.0],
        "max_log_odds": [5.0],

        # ScanMatcherParams
        "occ_thres": [1.0, 1.2, 1.6],
        "delta_r": [0.4, 0.5, 0.6],
        "surface_radius_m": [0.2],
        "min_free_ratio": [0.25],

        # ICPParams
        "max_n_points": [400],
        "neighbors_pca": [10],
        "max_iterations": [5],
        "max_correspondence_distance": [0.45, 0.6],
        "min_corresp": [15],
        "max_translation_jump": [0.3, 0.6],
        "max_rotation_jump_deg": [45.0],
        "max_acceptable_mean_error": [0.15],
    }

# def _grid_axes() -> Dict[str, List[Union[float, int]]]:
#     return {
#         # Playback sampling
#         "every_nth_beam_filter": [4],
#         "every_nth_beam_map": [2],

#         # OccupancyParams (OGM)
#         "increasing_probability": [0.7],
#         "decreasing_probability": [0.3],
#         "min_log_odds": [-5.0],
#         "max_log_odds": [5.0],

#         # ScanMatcherParams
#         "occ_thres": [0.8],
#         "delta_r": [0.5],
#         "surface_radius_m": [0.2],
#         "min_free_ratio": [0.25],

#         # ICPParams
#         "max_n_points": [400],
#         "neighbors_pca": [10],
#         "max_iterations": [5],
#         "max_correspondence_distance": [0.6],
#         "min_corresp": [15],
#         "max_translation_jump": [0.3],
#         "max_rotation_jump_deg": [45.0],
#         "max_acceptable_mean_error": [0.15],
#     }


def write_parameter_overview(
    path: str,
    n_repeats: int,
    start_pose: Tuple[float, float, float],
    override: bool = False,
) -> None:
    file_exists = ResultWriterScanMatching.create_path_and_check_if_file_exists(path=path)

    if file_exists and not override:
        print("\nParameter overview has not been saved because file already exists and override is set to False!")
        return

    axes = _grid_axes()
    example_params = next(generate_param_grid(start_pose=start_pose, n_repeats=1), None)

    payload = {
        "playback_dir": PLAYBACK_DIR,
        "playback_suffix": PLAYBACK_SUFFIX,
        "measurement_stddev": MEASUREMENT_STDDEV,
        "n_playback_steps": N_PLAYBACK_STEPS,
        "n_optimization_repeats": n_repeats,
        "seed_list": SEED_LIST,
        "start_pose": start_pose,
        "grid_axes": axes,
        "example_experiment_params": _to_jsonable(example_params) if example_params is not None else None,
    }

    with open(path, "w") as f:
        json.dump(_to_jsonable(payload), f, indent=2)

    print(f"\nParameter overview has been saved to:\n{path}")


def generate_param_grid(
    start_pose: Tuple[float, float, float],
    n_repeats: int = 1,
) -> Iterator[ExperimentParams]:
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    axes = _grid_axes()
    increasing_probs = axes["increasing_probability"]
    decreasing_probs = axes["decreasing_probability"]
    if len(increasing_probs) != len(decreasing_probs):
        raise ValueError(
            "increasing_probability and decreasing_probability must have the same length "
            "to be evaluated as paired values."
        )
    occupancy_prob_pairs = list(zip(increasing_probs, decreasing_probs))

    wheel_separation = _compute_wheel_separation()

    for repeat_idx in range(1, n_repeats + 1):
        for (
            every_nth_filter,
            every_nth_map,
            occupancy_prob_pair,
            min_log_odds,
            max_log_odds,
            occ_thres,
            delta_r,
            surface_radius_m,
            min_free_ratio,
            max_n_points,
            neighbors_pca,
            max_iterations,
            max_corr_dist,
            min_corresp,
            max_jump_trans,
            max_jump_rot_deg,
            max_acceptable_mean_error,
        ) in itertools.product(
            axes["every_nth_beam_filter"],
            axes["every_nth_beam_map"],
            occupancy_prob_pairs,
            axes["min_log_odds"],
            axes["max_log_odds"],
            axes["occ_thres"],
            axes["delta_r"],
            axes["surface_radius_m"],
            axes["min_free_ratio"],
            axes["max_n_points"],
            axes["neighbors_pca"],
            axes["max_iterations"],
            axes["max_correspondence_distance"],
            axes["min_corresp"],
            axes["max_translation_jump"],
            axes["max_rotation_jump_deg"],
            axes["max_acceptable_mean_error"],
        ):
            # Enforce integer-valued hyperparameters even if grid entries were provided as floats.
            every_nth_filter = int(every_nth_filter)
            every_nth_map = int(every_nth_map)
            max_n_points = int(max_n_points)
            neighbors_pca = int(neighbors_pca)
            max_iterations = int(max_iterations)
            min_corresp = int(min_corresp)

            increasing_probability, decreasing_probability = occupancy_prob_pair


            yield ExperimentParams(
                occupancy_params=OccupancyParams(
                    # All cells are initalized with this probability when map is initialized. 
                    prior_probability=0.5, 
                    # Min distance of the robot to the border of the map. If robot is closer than this to the border, the map will be extended.                   
                    min_distance_to_border=10.0,
                    # Cell is increased by the log Odds of this value when beam ends in this cell
                    increasing_probability=increasing_probability,
                    # All cells a beam passed will decreased by the log Odds of this value  
                    decreasing_probability=decreasing_probability,
                    # Max and Min possible log odds value a cell can have. 
                    min_log_odds=min_log_odds,
                    max_log_odds=max_log_odds,
                ),
                sensor_params=SensorParams(
                    min_sensor_range=MIN_SENSOR_RANGE,
                    max_sensor_range=MAX_SENSOR_RANGE,
                ),
                map_param=MapParameter(
                    map_width=10.0,
                    map_height=10.0,
                    grid_resolution_m=0.1,
                ),
                icp_params=ICPParams(
                    max_n_points=max_n_points,
                    max_correspondence_distance=max_corr_dist,
                    neighbors_pca=neighbors_pca,
                    max_iterations=max_iterations,
                    epsilon_rel=1e-3,
                    no_improvement_limit=3,
                    min_error=5e-4,
                    min_dtrans=1e-3,
                    min_drot=1e-2,
                    min_points=20,
                    min_corresp=min_corresp,
                    min_hessian_rank=3,
                    max_hessian_condition=1e8,
                    max_translation_jump=max_jump_trans,
                    max_rotation_jump=np.deg2rad(max_jump_rot_deg),
                    max_acceptable_mean_error=max_acceptable_mean_error,
                ),
                robot_params=RobotParams(
                    wheel_separation=wheel_separation,
                ),
                scan_matcher_params=ScanMatcherParams(
                    occ_thres=occ_thres,
                    delta_r=delta_r,
                    surface_radius_m=surface_radius_m,
                    min_free_ratio=min_free_ratio,
                ),
                particle_params=ParticleParams(
                    n_particles=1,
                    start_pose=start_pose,
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
                measurement_noise_stddev=MEASUREMENT_STDDEV,
                tag=(
                    f"nf{every_nth_filter}_nm{every_nth_map}_"
                    f"ip{increasing_probability}_dp{decreasing_probability}_"
                    f"lomin{min_log_odds}_lomax{max_log_odds}_"
                    f"ot{occ_thres}_dr{delta_r}_sr{surface_radius_m}_mfr{min_free_ratio}_"
                    f"mnp{max_n_points}_npca{neighbors_pca}_mi{max_iterations}_"
                    f"mcd{max_corr_dist}_mc{min_corresp}_mjt{max_jump_trans}_"
                    f"mjrd{max_jump_rot_deg}_mae{max_acceptable_mean_error}_"
                    f"rep{repeat_idx}"
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
        ensure_start_pose=True,
        prompt_for_missing_start_pose=True,
    )

    start_pose = tuple(raw_playback_data.metadata["robot_start_pose"])
    print(f"Using start pose for tuning: {start_pose}")

    playback_conv = PlaybackConverter()
    
    # Keep scans clean here. Measurement noise is injected per seed in the optimizer.
    playback_data = playback_conv.convert(
        raw_playback_data,
        measurement_stddev=None,
        min_range=MIN_SENSOR_RANGE,
        max_range=MAX_SENSOR_RANGE,
    )

    optimizer = build_optimizer()
    writer = ResultWriterScanMatching()

    # Store compact parameter overview (grid axes + one representative ExperimentParams)
    write_parameter_overview(
        path=PARAMETER_OVERVIEW_PATH,
        n_repeats=N_OPTIMIZATION_REPEATS,
        start_pose=start_pose,
        override=OVERRIDE_EXISTING_RESULTS,
    )

    ranked_runs = optimizer.optimize(
        playback_data=playback_data,
        param_grid=generate_param_grid(start_pose=start_pose, n_repeats=N_OPTIMIZATION_REPEATS),
        seeds=SEED_LIST,
        use_seed_list_for_measurement_noise=USE_SEED_LIST_FOR_MEASUREMENT_NOISE,
    )

    writer.write_summary_runs_csv(
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
