#!/usr/bin/env python3

import debugpy

import itertools
import json
import numpy as np
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union

from ..infrastructure.playback_loader import PlaybackLoader
from ..infrastructure.playback_converter import PlaybackConverter

from ..scan_matcher.warmp_up_numba_scan_matcher import warm_up_numba_scan_matcher

from ..rbpf.rbpf import RBPFFactory, ParticleParams, MotionModelParams, BeamRangeFinderMeasModelParams
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
from .aggregator_scanmatching import RankedRunConverterScanMatching, ResultAggregatorScanMatching
from .step_processor import StepProcessor


# Data path defs
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
STORAGE_DIR = DATA_DIR / "scan_matching" / "optimization_results"
PLAYBACK_DATA_DIR = DATA_DIR / "python_playback"

# Default storage
SUB_DIR = "sm_optm_7_1/"
OPTM_SUMMARY_PATH = STORAGE_DIR / SUB_DIR / "summary"
SCAN_MATCHING_STEP_TRACE_PATH = STORAGE_DIR / SUB_DIR / "trace_steps.csv"
PARAMETER_OVERVIEW_PATH = STORAGE_DIR / SUB_DIR / "params.json"

# Test storage
# SUB_DIR = "sm_test_3/"
# OPTM_SUMMARY_PATH = STORAGE_DIR / SUB_DIR / "summary"
# SCAN_MATCHING_STEP_TRACE_PATH = STORAGE_DIR / SUB_DIR / "trace_steps.csv"
# PARAMETER_OVERVIEW_PATH = STORAGE_DIR / SUB_DIR / "params.json"

# Ctrl debugger
DEBUG_CODE = False

# Switch between sequential and parallel optimization pipe
USE_PARALLEL_OPTM_PIPE = True

# Number of workers to use for multiprocessing tuning pipe
NUMBER_OF_WORKERS = 4
# Define whether to keep the step results or not. Don't keep for big grid search -> Too much memory!
KEEP_STEP_RESULTS = True

CSV_FLOAT_DECIMALS = 6
OVERRIDE_EXISTING_RESULTS = False
N_PLAYBACK_STEPS = None
N_OPTIMIZATION_REPEATS = 1
# SEED_LIST = [22, 23, 56]
SEED_LIST = [22, 56]

# Controls ONLY measurement-noise seeding behavior in optimizer:
# - True:  use values from SEED_LIST for deterministic per-seed measurement noise.
# - False: do not seed measurement noise (fresh random noise every run).
USE_SEED_LIST_FOR_MEASUREMENT_NOISE = True

# Define sttdev [m] to add noise to the playback measurement. This is only possible if the playback data doesnt include noise in the measurements.
MEASUREMENT_STDDEV = 0.03
MIN_SENSOR_RANGE = 0.1
MAX_SENSOR_RANGE = 10.0 

# Define whether ICP subsampling is skipped
ICP_SKIP_SUBSAMPLING = False

POSE_APPENDIX = ("x", "y", "theta_deg")

# Defines the columns of the step data that will be written to csv. All others in df will be skipped
STEP_COLS_TO_USE = [
    # General run information
    "rank",
    "score",
    "dataset_id",
    "map_name",
    "seed",
    "parameter_tag",
    "parameter_hash",

    "step_idx",
    "t",

    # Scan matching and ICP information
    "scan_match_failed",
    "icp_iterations",
    "n_correspondences",
    "use_transformation",
    "stop_reason",

    # Measurement and map information
    "n_measurements_total",
    "n_valid_measurements_filter",
    "n_valid_measurements_map_update",
    "n_map_points_extracted",

    "icp_best_trans_param",
    "icp_best_rot_abs_deg",
    "icp_mean_error",

    # Poses
    "true_pose_x",
    "true_pose_y",
    "true_pose_theta_deg",

    "raw_odom_pose_x",
    "raw_odom_pose_y",
    "raw_odom_pose_theta_deg",

    "pred_pose_x",
    "pred_pose_y",
    "pred_pose_theta_deg",

    "corr_pose_x",
    "corr_pose_y",
    "corr_pose_theta_deg",

    # Pose errors
    "raw_odom_trans_err",
    "pred_trans_err",
    "corr_trans_err",
    "raw_odom_rot_err_deg",
    "pred_rot_err_deg",
    "corr_rot_err_deg",

    "pred_to_corr_trans_err",
    "pred_to_corr_rot_err_deg",

    # Timings
    "t_ogm_ms",
    "t_scan_matching_ms",
    "t_prediction_ms",
    "t_map_extraction_ms",
    "t_correct_pose_ms",
]

# Defines a playback dataset, which includes the directory and suffix of the playback files.
@dataclass
class PlaybackDataset:
    playback_dir: Path
    playback_suffix: str


# Defines the playback data that is being used in the tuning pipe.
PLAYBACK_DATA_LIST = [
    # turtle bot map
    PlaybackDataset(
        playback_dir=PLAYBACK_DATA_DIR,
        playback_suffix="1779363559",
    ),
    # PlaybackDataset(
    #     playback_dir=PLAYBACK_DATA_DIR,
    #     playback_suffix="1779375646",
    # ),
    # AWS map
    PlaybackDataset(
        playback_dir=PLAYBACK_DATA_DIR,
        playback_suffix="1780397517",
    )
]


def debug():
    '''Starts debugpy and waits for debugger to attach.'''
    debugpy.listen(("localhost", 5678))
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()  



def _to_jsonable(value: Any) -> Any:
    '''Converts a value to a JSON-serializable format.'''
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


# Define params for big grid search after newly implemented grid based subsampling
def _grid_axes():
    '''
    Defines the parameter grid to use for the optimization process.
    '''
    return {
        "every_nth_beam_filter": [2],
        "every_nth_beam_map": [2],

        "occupancy_prob_pairs": [
            {
                "increasing_probability": 0.85,
                "decreasing_probability": 0.15,
            },
        ],
        "min_log_odds": [-5.0],
        "max_log_odds": [5.0],

        "occ_thres": [1.4],
        "delta_r": [0.6],
        "surface_radius_m": [0.2],
        "min_free_ratio": [0.4],

        "max_n_points": [800, 1200],
        "downsample_grid_size": [0.1],
        "neighbors_pca": [6],
        "max_iterations": [5],
        "max_correspondence_distance": [0.4],
        "min_corresp": [25],
        "max_translation_jump": [0.7],
        "max_rotation_jump_deg": [45.0],
        "max_acceptable_mean_error": [0.15],
    }


def write_parameter_overview(
    path: str,
    n_repeats: int,
    wheel_separation: float,
    override: bool = False,
) -> None:
    '''
    Writes the parameter grid together with all other experiment parameters to a JSON file for later reference.

    Parameters
    ----------
    path : str
        Path to the JSON file where the parameter overview will be saved.
    n_repeats : int
        Number of repeats for each parameter combination.
    wheel_separation : float
        Wheel separation of the robot, used in the motion model parameters.
    override : bool, optional
        If True, will override the existing file if it exists. Default is False.
    '''
    file_exists = ResultWriterScanMatching.create_path_and_check_if_file_exists(path=path)

    if file_exists and not override:
        print(f"\nParameter overview has not been saved because file already exists and override is set to False!\n{path}")
        return

    axes = _grid_axes()
    dummy_pose = None
    example_params = next(
        generate_param_grid(
            start_pose=dummy_pose,
            wheel_separation=wheel_separation,
            n_repeats=1,
        ),
        None,
    )
    example_params_json = (
        _to_jsonable(ScanMatchingOptimizer.generate_params_for_hash(example_params))
        if example_params is not None
        else None
    )

    payload = {
        # "playback_data_list": [asdict(ds) for ds in PLAYBACK_DATA_LIST],
        "measurement_stddev": MEASUREMENT_STDDEV,
        "n_playback_steps": N_PLAYBACK_STEPS,
        "n_optimization_repeats": n_repeats,
        "seed_list": SEED_LIST,
        "start_pose": dummy_pose,
        "grid_axes": axes,
        "example_experiment_params": example_params_json,
    }

    with open(path, "w") as f:
        json.dump(_to_jsonable(payload), f, indent=2)

    print(f"\nParameter overview has been saved to:\n{path}")


def generate_param_grid(
    start_pose: Tuple[float, float, float],
    wheel_separation: float,
    n_repeats: int = 1,
) -> Iterator[ExperimentParams]:
    '''
    Defined the parameter grid for the RBPF SLAM optimization. This is a generator that yields ExperimentParams for
    each combination of parameters in the grid.
    '''
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    axes = _grid_axes()

    occupancy_prob_sets = axes.get("occupancy_prob_pairs", [])
    occupancy_prob_pairs = []

    for i, prob_set in enumerate(occupancy_prob_sets):
        if not isinstance(prob_set, dict):
            raise TypeError(
                f"occupancy_prob_pairs[{i}] must be a dict, got {type(prob_set)}"
            )

        try:
            occupancy_prob_pairs.append(
                (
                    float(prob_set["increasing_probability"]),
                    float(prob_set["decreasing_probability"]),
                )
            )
        except KeyError as exc:
            raise KeyError(
                f"occupancy_prob_pairs[{i}] is missing required key: {exc}"
            ) from exc

    if not occupancy_prob_pairs:
        raise ValueError("No occupancy probability pairs configured.")

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
            downsample_grid_size,
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
            axes["downsample_grid_size"],
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
            downsample_grid_size = float(downsample_grid_size)
            skip_subsampling = bool(ICP_SKIP_SUBSAMPLING)
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
                    grid_resolution_m=0.05,
                ),
                icp_params=ICPParams(
                    max_n_points=max_n_points,
                    downsample_grid_size=downsample_grid_size,
                    skip_subsampling=skip_subsampling,
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
                measurement_model_params=BeamRangeFinderMeasModelParams(),
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
                    f"mnp{max_n_points}_dsgs{downsample_grid_size}_"
                    f"ss{int(skip_subsampling)}_"
                    f"npca{neighbors_pca}_mi{max_iterations}_"
                    f"mcd{max_corr_dist}_mc{min_corresp}_mjt{max_jump_trans}_"
                    f"mjrd{max_jump_rot_deg}_mae{max_acceptable_mean_error}_"
                    f"rep{repeat_idx}"
                ),
            )


def build_optimizer() -> ScanMatchingOptimizer:
    '''
    Builds and returns the Scan matcher optimizer and returns it. This function initializes the playback runner,
    evaluator, and optimizer with the necessary components.
    '''
    runner = PlaybackRunnerScanMatching(
        factory=RBPFFactory(),
        evaluator=ScanMatchingEvaluator(),
    )

    return ScanMatchingOptimizer(
        runner=runner,
        scorer=ScanMatchingScorer(),
    )



def scan_matcher_tuning_pipeline() -> None:
    '''
    Tuning pipe to find the best possible scan matcher parameter based on the defined parameter grid inside _grid_axes(). 
    Process:
        1) Defines storage
        2) initialize pipeline components
        3) Load each dataset and converts the loaded playback data
        4) Runs the optimizer that runs the scan matcher on all datasets with all parameter combinations and all seeds.
           Computes the metrics for each run and stores the results in a ranked list of runs.
        5) Sorts the estimated resutls by score
        6) Processes the ranked runs 
        7) Aggregates the summary results by dataset and parameter/seed combinations. 
        8) Stores the results of the pipeline
    '''
    # Define storage paths for results
    ranked_run_list = []
    ranked_scored_path = OPTM_SUMMARY_PATH.with_name(f"{OPTM_SUMMARY_PATH.name}_rank_scored.csv")
    agg_dataset_seed_path = OPTM_SUMMARY_PATH.with_name(f"{OPTM_SUMMARY_PATH.name}_agg_dataset_id_param.csv")
    agg_param_path = OPTM_SUMMARY_PATH.with_name(f"{OPTM_SUMMARY_PATH.name}_agg_param.csv")
    ranked_param_overview_path = OPTM_SUMMARY_PATH.with_name(f"{OPTM_SUMMARY_PATH.name}_ranked_param_overview.csv")

    # Initialize tuning pipe components
    playback_loader = PlaybackLoader()
    playback_conv = PlaybackConverter()
    optimizer = build_optimizer()
    writer = ResultWriterScanMatching()
    ranked_run_conv = RankedRunConverterScanMatching()
    result_aggregator = ResultAggregatorScanMatching()
    step_processor = StepProcessor()

    # Load each dataset and optimize with identical parameter/seed setup.
    optm_durations = []
    parameter_overview_written = False
    for playback_ds in PLAYBACK_DATA_LIST:
        print(
            f"\nLoading playback data:\n"
            f"suffix: {playback_ds.playback_suffix}\n"
            f"dir: {playback_ds.playback_dir}"
        )
        raw_playback_data = playback_loader.load(
            file_suffix=playback_ds.playback_suffix,
            filedir=playback_ds.playback_dir,
            n_steps=N_PLAYBACK_STEPS,
            ensure_start_pose=True,
            prompt_for_missing_start_pose=True,
        )

        # Extract start pose and wheel separation from playback metadata 
        start_pose = tuple(raw_playback_data.metadata["robot_start_pose"])
        wheel_separation = float(raw_playback_data.metadata["wheel_separation"])
        print(f"Using start pose for tuning: {start_pose}")
        print(f"Using wheel separation for tuning: {wheel_separation}")

        # Store compact parameter overview (grid axes + one representative ExperimentParams)
        if not parameter_overview_written:
            write_parameter_overview(
                path=PARAMETER_OVERVIEW_PATH,
                n_repeats=N_OPTIMIZATION_REPEATS,
                wheel_separation=wheel_separation,
                override=OVERRIDE_EXISTING_RESULTS,
            )
            parameter_overview_written = True

        # Keep scans clean here. Measurement noise is injected per seed in the optimizer.
        playback_data = playback_conv.convert(
            raw_playback_data,
            measurement_stddev=None,
            min_range=MIN_SENSOR_RANGE,
            max_range=MAX_SENSOR_RANGE,
        )

        # Run optimizer in sequential mode -> optimization results
        ranked_runs, optm_duration_s = optimizer.optimize(
            playback_data=playback_data,
            param_grid=generate_param_grid(
                start_pose=start_pose,
                wheel_separation=wheel_separation,
                n_repeats=N_OPTIMIZATION_REPEATS,
            ),
            seeds=SEED_LIST,
            dataset_id=playback_ds.playback_suffix,
            map_name=raw_playback_data.metadata.get("map", "unknown_map"),
            use_seed_list_for_measurement_noise=USE_SEED_LIST_FOR_MEASUREMENT_NOISE,
        )

        optm_durations.append(optm_duration_s)
        ranked_run_list.extend(ranked_runs)

    # Sort runs by score from lowest to highest across all datasets.
    ranked_run_list.sort(key=lambda ranked_run: ranked_run.score)

    cleaned_optm_duratios = [optm_dur_s for optm_dur_s in optm_durations if optm_dur_s is not None]
    if cleaned_optm_duratios is not None:
        overall_optm_duration_s = sum(cleaned_optm_duratios)
        print(f"\n\nFinished overall scan matching optimization in {overall_optm_duration_s} s")

    # Process step data into one flat DataFrame row per stored step if needed
    if KEEP_STEP_RESULTS:
        step_trace_df = step_processor.process_ranked_runs(
            ranked_runs=ranked_run_list,
            pose_appendix=POSE_APPENDIX,
        )

    # Aggregate results
    ranked_run_df = ranked_run_conv.to_dataframe(ranked_run_list)

    # Rank results by score
    rank_scored_df = result_aggregator.rank_by_score(
        ranked_run_df=ranked_run_df,
        score_col="score",
        ascending=True,
    )

    # Aggregate the summary results
    agg_dataset_seed_df = result_aggregator.aggregate_by_dataset_and_param(ranked_run_df)
    agg_param_df = result_aggregator.aggregate_by_params(agg_dataset_seed_df)
    ranked_param_overview_df = result_aggregator.build_ranked_parameter_overview(
        agg_param_df=agg_param_df,
        ranked_runs=ranked_run_list,
    )

    # Save results
    writer.write_dataframe_csv(
        path=ranked_scored_path,
        df=rank_scored_df,
        override=OVERRIDE_EXISTING_RESULTS,
        float_decimals=CSV_FLOAT_DECIMALS,
    )
    writer.write_dataframe_csv(
        path=agg_dataset_seed_path,
        df=agg_dataset_seed_df,
        override=OVERRIDE_EXISTING_RESULTS,
        float_decimals=CSV_FLOAT_DECIMALS,
    )
    writer.write_dataframe_csv(
        path=agg_param_path,
        df=agg_param_df,
        override=OVERRIDE_EXISTING_RESULTS,
        float_decimals=CSV_FLOAT_DECIMALS,
    )
    writer.write_dataframe_csv(
        path=ranked_param_overview_path,
        df=ranked_param_overview_df,
        override=OVERRIDE_EXISTING_RESULTS,
        float_decimals=CSV_FLOAT_DECIMALS,
    )

    # Save step data if needed
    if KEEP_STEP_RESULTS:
        writer.write_dataframe_csv(
            path=SCAN_MATCHING_STEP_TRACE_PATH,
            df=step_trace_df,
            override=OVERRIDE_EXISTING_RESULTS,
            float_decimals=CSV_FLOAT_DECIMALS,
            cols_to_use=STEP_COLS_TO_USE,
            label="Step trace DataFrame",
        )

    print("Scan-matching-only tuning run finished.")



def scan_matcher_tuning_pipeline_multiprocessing() -> None:
    '''
    Parallel tuning pipe to find the best possible scan matcher parameters based on the parameter grid defined in
    _grid_axes().
    Process:
        1) Defines storage
        2) Initializes pipeline components
        3) Loads each dataset and converts the loaded playback data
        4) Runs the optimizer in parallel on all datasets with all parameter combinations and all seeds.
           Computes the metrics for each run and stores the results in a ranked list of runs.
        5) Sorts the estimated results by score
        6) Processes the ranked runs
        7) Aggregates the summary results by dataset and parameter/seed combinations
        8) Stores the results of the pipeline
    '''
    # Define storage paths for results
    ranked_run_list = []
    ranked_scored_path = OPTM_SUMMARY_PATH.with_name(f"{OPTM_SUMMARY_PATH.name}_rank_scored.csv")
    agg_dataset_seed_path = OPTM_SUMMARY_PATH.with_name(f"{OPTM_SUMMARY_PATH.name}_agg_dataset_id_param.csv")
    agg_param_path = OPTM_SUMMARY_PATH.with_name(f"{OPTM_SUMMARY_PATH.name}_agg_param.csv")
    ranked_param_overview_path = OPTM_SUMMARY_PATH.with_name(f"{OPTM_SUMMARY_PATH.name}_ranked_param_overview.csv")

    # Initialize tuning pipe components
    playback_loader = PlaybackLoader()
    playback_conv = PlaybackConverter()
    optimizer = build_optimizer()
    writer = ResultWriterScanMatching()
    ranked_run_conv = RankedRunConverterScanMatching()
    result_aggregator = ResultAggregatorScanMatching()
    step_processor = StepProcessor()

    # Load each dataset and optimize with identical parameter/seed setup.
    optm_durations = []
    parameter_overview_written = False
    for playback_ds in PLAYBACK_DATA_LIST:
        print(
            f"\nLoading playback data:\n"
            f"suffix: {playback_ds.playback_suffix}\n"
            f"dir: {playback_ds.playback_dir}"
        )
        raw_playback_data = playback_loader.load(
            file_suffix=playback_ds.playback_suffix,
            filedir=playback_ds.playback_dir,
            n_steps=N_PLAYBACK_STEPS,
            ensure_start_pose=True,
            prompt_for_missing_start_pose=True,
        )

        # Extract start pose and wheel separation from playback metadata
        start_pose = tuple(raw_playback_data.metadata["robot_start_pose"])
        wheel_separation = float(raw_playback_data.metadata["wheel_separation"])
        print(f"Using start pose for tuning: {start_pose}")
        print(f"Using wheel separation for tuning: {wheel_separation}")

        # Store compact parameter overview (grid axes + one representative ExperimentParams)
        if not parameter_overview_written:
            write_parameter_overview(
                path=PARAMETER_OVERVIEW_PATH,
                n_repeats=N_OPTIMIZATION_REPEATS,
                wheel_separation=wheel_separation,
                override=OVERRIDE_EXISTING_RESULTS,
            )
            parameter_overview_written = True

        # Keep scans clean here. Measurement noise is injected per seed in the optimizer.
        playback_data = playback_conv.convert(
            raw_playback_data,
            measurement_stddev=None,
            min_range=MIN_SENSOR_RANGE,
            max_range=MAX_SENSOR_RANGE,
        )

        # Run optimizer in parallel
        ranked_runs, optm_duration_s = optimizer.optimize_parallel(
            playback_data=playback_data,
            param_grid=generate_param_grid(
                start_pose=start_pose,
                wheel_separation=wheel_separation,
                n_repeats=N_OPTIMIZATION_REPEATS,
            ),
            seeds=SEED_LIST,
            dataset_id=playback_ds.playback_suffix,
            map_name=raw_playback_data.metadata.get("map", "unknown_map"),
            use_seed_list_for_measurement_noise=USE_SEED_LIST_FOR_MEASUREMENT_NOISE,
            max_workers=NUMBER_OF_WORKERS,
            keep_step_results=KEEP_STEP_RESULTS,
        )

        optm_durations.append(optm_duration_s)
        ranked_run_list.extend(ranked_runs)

    # Sort runs by score from lowest to highest across all datasets.
    ranked_run_list.sort(key=lambda ranked_run: ranked_run.score)

    cleaned_optm_duratios = [optm_dur_s for optm_dur_s in optm_durations if optm_dur_s is not None]
    if cleaned_optm_duratios is not None:
        overall_optm_duration_s = sum(cleaned_optm_duratios)
        print(f"\n\nFinished overall scan matching optimization in {overall_optm_duration_s} s")

    # Process step data into one flat DataFrame row per stored step if needed
    if KEEP_STEP_RESULTS:
        step_trace_df = step_processor.process_ranked_runs(
            ranked_runs=ranked_run_list,
            pose_appendix=POSE_APPENDIX,
        )

    # Aggregate results
    ranked_run_df = ranked_run_conv.to_dataframe(ranked_run_list)

    # Rank results by score
    rank_scored_df = result_aggregator.rank_by_score(
        ranked_run_df=ranked_run_df,
        score_col="score",
        ascending=True,
    )

    # Aggregate the summary results
    agg_dataset_seed_df = result_aggregator.aggregate_by_dataset_and_param(ranked_run_df)
    agg_param_df = result_aggregator.aggregate_by_params(agg_dataset_seed_df)
    ranked_param_overview_df = result_aggregator.build_ranked_parameter_overview(
        agg_param_df=agg_param_df,
        ranked_runs=ranked_run_list,
    )

    # Save results
    writer.write_dataframe_csv(
        path=ranked_scored_path,
        df=rank_scored_df,
        override=OVERRIDE_EXISTING_RESULTS,
        float_decimals=CSV_FLOAT_DECIMALS,
    )
    writer.write_dataframe_csv(
        path=agg_dataset_seed_path,
        df=agg_dataset_seed_df,
        override=OVERRIDE_EXISTING_RESULTS,
        float_decimals=CSV_FLOAT_DECIMALS,
    )
    writer.write_dataframe_csv(
        path=agg_param_path,
        df=agg_param_df,
        override=OVERRIDE_EXISTING_RESULTS,
        float_decimals=CSV_FLOAT_DECIMALS,
    )
    writer.write_dataframe_csv(
        path=ranked_param_overview_path,
        df=ranked_param_overview_df,
        override=OVERRIDE_EXISTING_RESULTS,
        float_decimals=CSV_FLOAT_DECIMALS,
    )
    
    # Save step data if needed
    if KEEP_STEP_RESULTS:
        writer.write_dataframe_csv(
            path=SCAN_MATCHING_STEP_TRACE_PATH,
            df=step_trace_df,
            override=OVERRIDE_EXISTING_RESULTS,
            float_decimals=CSV_FLOAT_DECIMALS,
            cols_to_use=STEP_COLS_TO_USE,
            label="Step trace DataFrame",
        )

    print("Scan-matching-only tuning run finished.")



def main() -> None:
    # Start Debugger 
    if DEBUG_CODE:
        debug()

    # Warm up numba functions
    warm_up_numba_scan_matcher()

    if USE_PARALLEL_OPTM_PIPE:
        # Scan matcher tuning pipe parallel
        scan_matcher_tuning_pipeline_multiprocessing()
    else:
        # Scan matcher unting pipe sequential
        scan_matcher_tuning_pipeline()

    


if __name__ == "__main__":
    main()
