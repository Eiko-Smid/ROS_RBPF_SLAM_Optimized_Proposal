#!/usr/bin/env python3

import debugpy

import itertools
import json
import numpy as np
from dataclasses import asdict, dataclass, is_dataclass

from .playback_defs import ExperimentParams, PlaybackData
# from .playback_loader import load_playback_dataset

from ..infrastructure.playback_loader import PlaybackLoader
from ..infrastructure.playback_converter import PlaybackConverter

from ..rbpf.rbpf import (
    RBPFFactory,
    ParticleParams,
    MotionModelParams,
    MeasurementModelParams,
    BeamRangeFinderMeasModelParams
)
from ..rbpf.scan_match_factory import (
    OccupancyParams,
    SensorParams,
    MapParameter,
    ICPParams,
    RobotParams,
    ScanMatcherParams,
    ScanMatchFactory
)

from ..scan_matcher.icp_scan_matching import warmup_numba_functions

# from .evaluator import RBPFEvaluator
from .evaluator_mult_part import RBPFEValMultParticles as RBPFEvaluator
from .playback_runner import PlaybackRunner, RawOdometryPropagator
from .scorer import RunScorer
from .optimizer import RBPFOptimizer
from .result_writer import ResultWriter
from .aggregator import RankedRunConverter, ResultAggregator
from .step_processor import StepProcessor


'''

1. Analyze RBPF with multiple particles on new tuning pipeline


    1.1 Weight corr for particle poes = mu


    1.2 Weight corr for particle pose = sample from proposal


    1.3 same as 1.2 but with limited proposal uncertainty


    1.4 Grid run with different scale and fixed limit values of cov

    1.5 Same as 46_4 but with 30 particles

        -   This is the max number of particles I can use to be fast enough for real time slam
        -

    1.6 run with best 3 obtained scales and max values and adapting limit here

        Goal: See if different min values can fix our problem

        Result:
            - Seems like this really worked
            - Because we ensure enough particle diversity when we don't scale the values too low! 


    1.7 Test best configurations on all maps/seeds

        -   Now we wanne test if the configurtaions we found are stabel on all maps and seeds
        -   If not we can't use them!


    1.8 Adapt scaling again without changing min/max limits of proposal downscaling
        -   We now only test it with different scaling and different neff thresholds on two different maps and seeds


    1.9 do same experiemnt as 1.8 but dont ignore the cov in cov-matrix downsampling 
        - Before we only extracted the std values of the cov and scaled this
        - THis time we scaled the whole cov matrix including the cov values of the off-diagonal elements
        - These define the shape by the mahalanobis distance and therefore the shape of the proposal distribution
        - So we also take care of teh orientation of the proposal distribution 

        
    1.10 Same computation as before but this time we used valid angles in mu of optimized proposal


    1.11 Delete valid angles verify new candidate from 1.9 on all maps
    
        -   Since the valid angle in the proposal mu changed the results we deleted it for the moment
        -   Candidate's:
                n_particles = 30
                cov_std_scale = [0.5, 0.6]
                neff_thres_ratio = [0.3]
                cov_max_std_xy = 0.02
                cov_max_std_theta = np.deg2rad(1.15)
                min_std_xy = [0.0, 0.001]
                min_std_theta = [0.0, np.deg2rad(0.03)]

        - This time we added max and min limits for proposal again


    1.12 Also validation run but this time without min/max limits for proposal downscaling


    1.13 Same validation run as 1.12 but maps are getting stored


    
    1.1N Same as 1.12 but with higher max trans jump (TODO)
        - Not done yet!

'''


# Playback data path defs
STORAGE_DIR = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optm_results_mult_part/"

# Default storage
# SUB_DIR = "proposal_optm_1_14/"
# OPTM_SUMMARY_PATH= STORAGE_DIR + SUB_DIR + 'summary'
# STEP_TRACE_PATH = STORAGE_DIR + SUB_DIR + 'steps.csv'
# PROPOSAL_WEIGHTS_PATH = STORAGE_DIR + SUB_DIR + 'proposal_weights.csv'
# PARAMETER_OVERVIEW_PATH = STORAGE_DIR + SUB_DIR + 'params.json'
# RUN_STORAGE_DIR = STORAGE_DIR + SUB_DIR + 'runs/'

# Test storage
SUB_DIR = "proposal_optm_test_4_linear_pipe/"
OPTM_SUMMARY_PATH= STORAGE_DIR + SUB_DIR + 'summary'
STEP_TRACE_PATH = STORAGE_DIR + SUB_DIR + 'steps.csv'
PROPOSAL_WEIGHTS_PATH = STORAGE_DIR + SUB_DIR + 'proposal_weights.csv'
PARAMETER_OVERVIEW_PATH = STORAGE_DIR + SUB_DIR + 'params.json'
RUN_STORAGE_DIR = STORAGE_DIR + SUB_DIR + 'runs/'

USED_MEAS_MODEL = "LaserRangeFinderModel"
# USED_MEAS_MODEL = "NN_Based_Gmap_Probs"
# USED_MEAS_MODEL = "GMAPPING"

# Number of workers to use for multiprocessing tuning pipe
NUMBER_OF_WORKERS = 4
# Define whether to keep the step results or not. Don't keep for big grid search -> Too much memory!
KEEP_STEP_RESULTS = True
STORE_MAP_DATA = True
CSV_FLOAT_DECIMALS = 6

OVERRIDE_EXISTING_RESULTS = False
N_PLAYBACK_STEPS = 50             # Set an integer (e.g. 200) to use only the first N steps. None = all steps are used.
N_OPTIMIZATION_REPEATS = 1          # Number of full grid passes. 3 means each parameter combination is evaluated three times.
# SEED_LIST = [22, 23, 56]
SEED_LIST = [22, 56]
# SEED_LIST = [22]

# Controls ONLY measurement-noise seeding behavior in optimizer:
# - True:  use values from SEED_LIST for deterministic per-seed measurement noise.
# - False: do not seed measurement noise (fresh random noise every run).
USE_SEED_LIST_FOR_MEASUREMENT_NOISE = True

# Define sttdev [m] to add noise to the playback measurements.
# Set to None to disable noise injection.
MEASUREMENT_STDDEV = 0.03
MIN_SENSOR_RANGE = 0.1
MAX_SENSOR_RANGE = 10.0

POSE_APPENDIX = ("x", "y", "theta")

STEP_COLS_TO_USE = [
    # General information
    "rank",
    "score",
    "dataset_id",
    "map_name",
    "seed",
    "parameter_tag",
    "parameter_hash",

    "step_idx",
    "t",
    "t_step_duration",

    # Scan match info
    "scan_match_failed",
    "scan_match_failed_fallback",

    # Poses
    "true_pose_x",
    "true_pose_y",
    "true_pose_theta",

    "raw_odom_pose_x",
    "raw_odom_pose_y",
    "raw_odom_pose_theta",

    "weighted_mean_pose_x",
    "weighted_mean_pose_y",
    "weighted_mean_pose_theta",

    "best_particle_pose_x",
    "best_particle_pose_y",
    "best_particle_pose_theta",
    "best_particle_weight",

    "closest_particle_pose_before_resampling_x",
    "closest_particle_pose_before_resampling_y",
    "closest_particle_pose_before_resampling_theta",

    "closest_particle_pose_after_resampling_x",
    "closest_particle_pose_after_resampling_y",
    "closest_particle_pose_after_resampling_theta",

    "map_traj_x",
    "map_traj_y",
    "map_traj_theta",

    # Std of weighted particle pose
    "weighted_part_std_x",
    "weighted_part_std_y",
    "weighted_part_std_theta",

    # Pose errors
    # trans
    "trans_err_raw_odom",
    "trans_err_weighted_mean",
    "trans_err_best_particle",
    "trans_err_closest_p_before_resampling",
    "trans_err_closest_p_after_resampling",
    "trans_err_map_traj",

    # rot
    "rot_err_raw_odom", 
    "rot_err_weighted_mean",
    "rot_err_best_particle",
    "rot_err_closest_p_before_resampling",
    "rot_err_closest_p_after_resampling",
    "rot_err_map_traj",

    # Resampling
    "neff",
    "resampling",

]

SUMMARY_COLS_TO_EXCLUDE = [
    # Proposal counters
    "meas_model_prop_valid_beam_count",
    "meas_model_prop_map_hit_count",
    "meas_model_prop_no_map_hit_count",
    "meas_model_prop_out_of_map_count",
    "meas_model_prop_unknown_ray_count",
    "meas_model_prop_known_free_ray_count",
    "meas_model_prop_unexpected_known_free_count",

    # Fallback counters
    "meas_model_fallback_valid_beam_count",
    "meas_model_fallback_map_hit_count",
    "meas_model_fallback_no_map_hit_count",
    "meas_model_fallback_out_of_map_count",
    "meas_model_fallback_unknown_ray_count",
    "meas_model_fallback_known_free_ray_count",
    "meas_model_fallback_unexpected_known_free_count",
]



@dataclass
class PlaybackDataset:
    playback_dir: str
    playback_suffix: str


# # Define datasets to load
# PLAYBACK_DATA_LIST = [
#     # Turtle bot map
#     PlaybackDataset(
#         playback_dir="/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/python_playback/",
#         playback_suffix="1779363559",
#     ), 
#     # AWS indoor map
#     PlaybackDataset(
#         playback_dir="/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/python_playback/",
#         playback_suffix="1780397517",
#     ),    
# ]

# PLAYBACK_DATA_LIST = [
#     # Turtle bot map
#     # PlaybackDataset(
#     #     playback_dir="/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/python_playback/",
#     #     playback_suffix="1781885725",
#     # ),
#     # AWS indoor map
#     PlaybackDataset(
#         playback_dir="/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/python_playback/",
#         playback_suffix="1781885274",
#     ), 
#     # AWS bookstore map   
#     PlaybackDataset(
#         playback_dir="/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/python_playback/",
#         playback_suffix="1782917349",
#     )
# ]

# Evaluation data
# PLAYBACK_DATA_LIST = [
#     # AWS bookstore map   
#     PlaybackDataset(
#         playback_dir="/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/python_playback/",
#         playback_suffix="1782917349",
#     ),
#     # Turtle bot map unsee area
#     PlaybackDataset(
#         playback_dir="/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/python_playback/",
#         playback_suffix="1783013274",
#     ),
#     # Cafe map
#     PlaybackDataset(
#         playback_dir="/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/python_playback/",
#         playback_suffix="1783013816",
#     ),
#     # AWS indoor map different path, same area
#     PlaybackDataset(
#         playback_dir="/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/python_playback/",
#         playback_suffix="1783014916",
#     ),
# ]


# All maps (Train + eval)
PLAYBACK_DATA_LIST = [
    # Turtle bot map
    PlaybackDataset(
        playback_dir="/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/python_playback/",
        playback_suffix="1781885725",
    ), 
    # AWS indoor map
    PlaybackDataset(
        playback_dir="/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/python_playback/",
        playback_suffix="1781885274",
    ),
    # # Turtle bot map unsee area
    # PlaybackDataset(
    #     playback_dir="/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/python_playback/",
    #     playback_suffix="1783013274",
    # ),
    # # AWS indoor map different path, same area
    # PlaybackDataset(
    #     playback_dir="/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/python_playback/",
    #     playback_suffix="1783014916",
    # ),
    # # Cafe map
    # PlaybackDataset(
    #     playback_dir="/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/python_playback/",
    #     playback_suffix="1783013816",
    # ),
    # # AWS bookstore map   
    # PlaybackDataset(
    #     playback_dir="/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/python_playback/",
    #     playback_suffix="1782917349",
    # ),
]



# Bookstore map
# PLAYBACK_DATA_LIST = [
#     PlaybackDataset(
#         playback_dir="/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/python_playback/",
#         playback_suffix="1782917349",
#     )
# ]


def _to_jsonable(value):
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



# def _grid_axes() -> dict:
#     return {
#         # General rbpf params
#         "every_nth_beam_filter": [2],               # use every nth beam for proposal/scan matching
#         "every_nth_beam_map": [2],                  # use every nth beam for map update
#         "n_particles": [30],                    # number of particles in the RBPF
#         "neff_threshold": [None],                     # Number of effective particles threshold for resampling

#         # Measurement model params
#         "sigma_measurement": [0.06],                # measurement uncertainty [m]
#         "meas_kernel_size": [1],                    # Define search space size around beam endpoint for gmapping like measurement likelihood
        
#         # Beam range finder measurement model params
#         "beam_occ_thresh": [1.4],
#         "beam_free_thresh": [-1.4],
#         "beam_unknown_thresh": [0.3],
#         "beam_known_free_ratio_thresh": [0.7],

#         "beam_model_param_sets": [
#             {
#                 "beam_w_hit": 0.50,
#                 "beam_w_short": 0.30,
#                 "beam_lambda_short": 0.20,
#                 "beam_w_max": 0.10,
#                 "beam_w_rand": 0.10,
#             },
#         ],

#         "beam_extra_param_sets": [
#                # Candidate B measurement region
#             {
#                 "beam_sigma_hit": 0.07,
#                 "beam_alpha_meas": 0.075,
#                 "beam_p_unknown": 0.10,
#                 "beam_p_out_of_map": 0.15,
#                 "beam_p_unexpected_known_free": 0.00,
#                 "beam_p_pred_below_min": 0.02,
#                 "beam_step": 2,
#             },       
#         ],
        
#         # Motion model params
#         "sigma_xy_motion": [0.12],            # motion model uncertainty in x and y direction [m]
#         "sigma_theta": [0.11],                      # motion model uncertainty in theta direction [rad]
#         "ctrl_motion_fac": [0.1],                   # control motion factor for translational movement under uncertainty
#         "ctrl_turn_fac": [0.15],                    # control turn factor for rotational movement under uncertainty
        
#         # Proposal params (bound sets).
#         # Each dict is one fixed combination of:
#         # proposal_sigma_xy, proposal_sigma_theta, n_samples_dir
#         # so these three values are sampled together (no Cartesian product among them).
#         "proposal_param_sets": [
#             {
#                 "proposal_sigma_xy": 0.06,      # Proposal window size in x/y direction [m]
#                 "proposal_sigma_theta": 0.025,   # proposal window size in theta direction [rad]
#                 "n_samples_dir": 3,             # samples per direction for proposal sampling (total samples = n_samples_dir^3)
#             }
#         ],
#         # Proposal covariance scale/limit params (bound sets).
#         # Each dict is one fixed combination propagated to sample_from_proposal_limit.
#         "scale_limit_cov": [
#             {
#                 "cov_std_scale": 0.25,
#                 "cov_max_std_xy": 0.020,
#                 "cov_max_std_theta": np.deg2rad(1.15),
#                 "min_std_xy": 0.0,
#                 "min_std_theta": 0.0,
#             },
#             {
#                 "cov_std_scale": 0.35,
#                 "cov_max_std_xy": 0.020,
#                 "cov_max_std_theta": np.deg2rad(1.15),
#                 "min_std_xy": 0.0,
#                 "min_std_theta": 0.0,
#             },
#             {
#                 "cov_std_scale": 0.50,
#                 "cov_max_std_xy": 0.020,
#                 "cov_max_std_theta": np.deg2rad(1.15),
#                 "min_std_xy": 0.0,
#                 "min_std_theta": 0.0,
#             },
#             {
#                 "cov_std_scale": 0.75,
#                 "cov_max_std_xy": 0.020,
#                 "cov_max_std_theta": np.deg2rad(1.15),
#                 "min_std_xy": 0.0,
#                 "min_std_theta": 0.0,
#             },
#         ],
#         # TODO: Delete proposal values when no longer needed later on
#         "proposal_alpha": [1.0],
#         "proposal_beta": [1.0],

#         # ScanMatcherParams (map extraction)
#         "surface_radius_m": [0.2],      # TODO: Later change the name cause we search in a quadratic window not in circle
#         "min_free_ratio": [0.4],

#         # ICP jump thresholds
#         "max_translation_jump": [0.7],
#         "max_rotation_jump_deg": [45.0],
#     }



# Estimate good min values
# def _grid_axes() -> dict:
#     return {
#         # General rbpf params
#         "every_nth_beam_filter": [2],               # use every nth beam for proposal/scan matching
#         "every_nth_beam_map": [2],                  # use every nth beam for map update
#         # "n_particles": [15, 25, 30],                    # number of particles in the RBPF
#         "neff_threshold": [None],                     # Number of effective particles threshold for resampling

#         # Measurement model params
#         "sigma_measurement": [0.06],                # measurement uncertainty [m]
#         "meas_kernel_size": [1],                    # Define search space size around beam endpoint for gmapping like measurement likelihood
        
#         # Beam range finder measurement model params
#         "beam_occ_thresh": [1.4],
#         "beam_free_thresh": [-1.4],
#         "beam_unknown_thresh": [0.3],
#         "beam_known_free_ratio_thresh": [0.7],

#         "beam_model_param_sets": [
#             {
#                 "beam_w_hit": 0.50,
#                 "beam_w_short": 0.30,
#                 "beam_lambda_short": 0.20,
#                 "beam_w_max": 0.10,
#                 "beam_w_rand": 0.10,
#             },
#         ],

#         "beam_extra_param_sets": [
#                # Candidate B measurement region
#             {
#                 "beam_sigma_hit": 0.07,
#                 "beam_alpha_meas": 0.075,
#                 "beam_p_unknown": 0.10,
#                 "beam_p_out_of_map": 0.15,
#                 "beam_p_unexpected_known_free": 0.00,
#                 "beam_p_pred_below_min": 0.02,
#                 "beam_step": 2,
#             },       
#         ],
        
#         # Motion model params
#         "sigma_xy_motion": [0.12],            # motion model uncertainty in x and y direction [m]
#         "sigma_theta": [0.11],                      # motion model uncertainty in theta direction [rad]
#         "ctrl_motion_fac": [0.1],                   # control motion factor for translational movement under uncertainty
#         "ctrl_turn_fac": [0.15],                    # control turn factor for rotational movement under uncertainty
        
#         # Proposal params (bound sets).
#         # Each dict is one fixed combination of:
#         # proposal_sigma_xy, proposal_sigma_theta, n_samples_dir
#         # so these three values are sampled together (no Cartesian product among them).
#         "proposal_param_sets": [
#             {
#                 "proposal_sigma_xy": 0.06,      # Proposal window size in x/y direction [m]
#                 "proposal_sigma_theta": 0.025,   # proposal window size in theta direction [rad]
#                 "n_samples_dir": 3,             # samples per direction for proposal sampling (total samples = n_samples_dir^3)
#             }
#         ],
#         # Proposal covariance scale/limit params (bound sets).
#         # Each dict is one fixed combination propagated to sample_from_proposal_limit.

#         "scale_limit_cov": [
#             # ============================================================
#             # cov_std_scale = 0.35
#             # ============================================================
#             {
#                 "n_particles": 30,
#                 "cov_std_scale": 0.35,
#                 "cov_max_std_xy": 0.020,
#                 "cov_max_std_theta": np.deg2rad(1.15),
#                 "min_std_xy": 0.001,
#                 "min_std_theta": np.deg2rad(0.03),
#             },
#             {
#                 "n_particles": 30,
#                 "cov_std_scale": 0.35,
#                 "cov_max_std_xy": 0.020,
#                 "cov_max_std_theta": np.deg2rad(1.15),
#                 "min_std_xy": 0.002,
#                 "min_std_theta": np.deg2rad(0.05),
#             },
#             {
#                 "n_particles": 30,
#                 "cov_std_scale": 0.35,
#                 "cov_max_std_xy": 0.020,
#                 "cov_max_std_theta": np.deg2rad(1.15),
#                 "min_std_xy": 0.003,
#                 "min_std_theta": np.deg2rad(0.08),
#             },
#             {
#                 "n_particles": 30,
#                 "cov_std_scale": 0.35,
#                 "cov_max_std_xy": 0.020,
#                 "cov_max_std_theta": np.deg2rad(1.15),
#                 "min_std_xy": 0.005,
#                 "min_std_theta": np.deg2rad(0.10),
#             },

#             # ============================================================
#             # cov_std_scale = 0.50
#             # ============================================================
#             {
#                 "n_particles": 25,
#                 "cov_std_scale": 0.50,
#                 "cov_max_std_xy": 0.020,
#                 "cov_max_std_theta": np.deg2rad(1.15),
#                 "min_std_xy": 0.001,
#                 "min_std_theta": np.deg2rad(0.03),
#             },
#             {
#                 "n_particles": 25,
#                 "cov_std_scale": 0.50,
#                 "cov_max_std_xy": 0.020,
#                 "cov_max_std_theta": np.deg2rad(1.15),
#                 "min_std_xy": 0.002,
#                 "min_std_theta": np.deg2rad(0.05),
#             },
#             {
#                 "n_particles": 25,
#                 "cov_std_scale": 0.50,
#                 "cov_max_std_xy": 0.020,
#                 "cov_max_std_theta": np.deg2rad(1.15),
#                 "min_std_xy": 0.003,
#                 "min_std_theta": np.deg2rad(0.08),
#             },
#             {
#                 "n_particles": 25,
#                 "cov_std_scale": 0.50,
#                 "cov_max_std_xy": 0.020,
#                 "cov_max_std_theta": np.deg2rad(1.15),
#                 "min_std_xy": 0.005,
#                 "min_std_theta": np.deg2rad(0.10),
#             },

#             # ============================================================
#             # cov_std_scale = 0.25
#             # ============================================================
#             {
#                 "n_particles": 15,
#                 "cov_std_scale": 0.25,
#                 "cov_max_std_xy": 0.020,
#                 "cov_max_std_theta": np.deg2rad(1.15),
#                 "min_std_xy": 0.001,
#                 "min_std_theta": np.deg2rad(0.03),
#             },
#             {
#                 "n_particles": 15,
#                 "cov_std_scale": 0.25,
#                 "cov_max_std_xy": 0.020,
#                 "cov_max_std_theta": np.deg2rad(1.15),
#                 "min_std_xy": 0.002,
#                 "min_std_theta": np.deg2rad(0.05),
#             },
#             {
#                 "n_particles": 15,
#                 "cov_std_scale": 0.25,
#                 "cov_max_std_xy": 0.020,
#                 "cov_max_std_theta": np.deg2rad(1.15),
#                 "min_std_xy": 0.003,
#                 "min_std_theta": np.deg2rad(0.08),
#             },
#             {
#                 "n_particles": 15,
#                 "cov_std_scale": 0.25,
#                 "cov_max_std_xy": 0.020,
#                 "cov_max_std_theta": np.deg2rad(1.15),
#                 "min_std_xy": 0.005,
#                 "min_std_theta": np.deg2rad(0.10),
#             },
#         ],
#         # TODO: Delete proposal values when no longer needed later on
#         "proposal_alpha": [1.0],
#         "proposal_beta": [1.0],

#         # ScanMatcherParams (map extraction)
#         "surface_radius_m": [0.2],      # TODO: Later change the name cause we search in a quadratic window not in circle
#         "min_free_ratio": [0.4],

#         # ICP jump thresholds
#         "max_translation_jump": [0.7],
#         "max_rotation_jump_deg": [45.0],
#     }

# Validate best candidates against all maps
# def _grid_axes() -> dict:
#     return {
#         # General rbpf params
#         "every_nth_beam_filter": [2],               # use every nth beam for proposal/scan matching
#         "every_nth_beam_map": [2],                  # use every nth beam for map update
#         "n_particles": [30],                    # number of particles in the RBPF
#         "neff_thres_ratio": [0.3],             # neff threshold as ratio of n_particles

#         # Measurement model params
#         "sigma_measurement": [0.06],                # measurement uncertainty [m]
#         "meas_kernel_size": [1],                    # Define search space size around beam endpoint for gmapping like measurement likelihood
        
#         # Beam range finder measurement model params
#         "beam_occ_thresh": [1.4],
#         "beam_free_thresh": [-1.4],
#         "beam_unknown_thresh": [0.3],
#         "beam_known_free_ratio_thresh": [0.7],

#         "beam_model_param_sets": [
#             {
#                 "beam_w_hit": 0.50,
#                 "beam_w_short": 0.30,
#                 "beam_lambda_short": 0.20,
#                 "beam_w_max": 0.10,
#                 "beam_w_rand": 0.10,
#             },
#         ],

#         "beam_extra_param_sets": [
#                # Candidate B measurement region
#             {
#                 "beam_sigma_hit": 0.07,
#                 "beam_alpha_meas": 0.075,
#                 "beam_p_unknown": 0.10,
#                 "beam_p_out_of_map": 0.15,
#                 "beam_p_unexpected_known_free": 0.00,
#                 "beam_p_pred_below_min": 0.02,
#                 "beam_step": 2,
#             },       
#         ],
        
#         # Motion model params
#         "sigma_xy_motion": [0.12],                  # motion model uncertainty in x and y direction [m]
#         "sigma_theta": [0.11],                      # motion model uncertainty in theta direction [rad]
#         "ctrl_motion_fac": [0.1],                   # control motion factor for translational movement under uncertainty
#         "ctrl_turn_fac": [0.15],                    # control turn factor for rotational movement under uncertainty
        
#         # Proposal params (bound sets).
#         # Each dict is one fixed combination of:
#         # proposal_sigma_xy, proposal_sigma_theta, n_samples_dir
#         # so these three values are sampled together (no Cartesian product among them).
#         "proposal_param_sets": [
#             {
#                 "proposal_sigma_xy": 0.06,      # Proposal window size in x/y direction [m]
#                 "proposal_sigma_theta": 0.025,   # proposal window size in theta direction [rad]
#                 "n_samples_dir": 3,             # samples per direction for proposal sampling (total samples = n_samples_dir^3)
#             }
#         ],
#         # Proposal covariance scale/limit params (bound sets).
#         # Each dict is one fixed combination propagated to sample_from_proposal_limit.

#         "scale_limit_cov": [
#             # {
#             #     "cov_std_scale": 0.9,
#             #     "cov_max_std_xy": 1.0,
#             #     "cov_max_std_theta": np.deg2rad(10),
#             #     "min_std_xy": 0.0,
#             #     "min_std_theta": np.deg2rad(0.0),
#             # },
#             # {
#             #     "cov_std_scale": 0.8,
#             #     "cov_max_std_xy": 1.0,
#             #     "cov_max_std_theta": np.deg2rad(10),
#             #     "min_std_xy": 0.0,
#             #     "min_std_theta": np.deg2rad(0.0),
#             # },
#             # {
#             #     "cov_std_scale": 0.7,
#             #     "cov_max_std_xy": 1.0,
#             #     "cov_max_std_theta": np.deg2rad(10),
#             #     "min_std_xy": 0.0,
#             #     "min_std_theta": np.deg2rad(0.0),
#             # },
#             {
#                 "cov_std_scale": 0.5,
#                 "cov_max_std_xy": 1.0,
#                 "cov_max_std_theta": np.deg2rad(10),
#                 "min_std_xy": 0.0,
#                 "min_std_theta": np.deg2rad(0.0),
#             },
#             {
#                 "cov_std_scale": 0.6,
#                 "cov_max_std_xy": 1.0,
#                 "cov_max_std_theta": np.deg2rad(10),
#                 "min_std_xy": 0.0,
#                 "min_std_theta": np.deg2rad(0.0),
#             },

#         ],
#         # TODO: Delete proposal values when no longer needed later on
#         "proposal_alpha": [1.0],
#         "proposal_beta": [1.0],

#         # ScanMatcherParams (map extraction)
#         "surface_radius_m": [0.2],      # TODO: Later change the name cause we search in a quadratic window not in circle
#         "min_free_ratio": [0.4],

#         # ICP jump thresholds
#         "max_translation_jump": [0.7],
#         "max_rotation_jump_deg": [45.0],
#     }


def _grid_axes() -> dict:
    return {
        # General rbpf params
        "every_nth_beam_filter": [2],               # use every nth beam for proposal/scan matching
        "every_nth_beam_map": [2],                  # use every nth beam for map update
        "n_particles": [20],                    # number of particles in the RBPF
        "neff_thres_ratio": [0.3],             # neff threshold as ratio of n_particles

        # Measurement model params
        "sigma_measurement": [0.06],                # measurement uncertainty [m]
        "meas_kernel_size": [1],                    # Define search space size around beam endpoint for gmapping like measurement likelihood
        
        # Beam range finder measurement model params
        "beam_occ_thresh": [1.4],
        "beam_free_thresh": [-1.4],
        "beam_unknown_thresh": [0.3],
        "beam_known_free_ratio_thresh": [0.7],

        "beam_model_param_sets": [
            {
                "beam_w_hit": 0.50,
                "beam_w_short": 0.30,
                "beam_lambda_short": 0.20,
                "beam_w_max": 0.10,
                "beam_w_rand": 0.10,
            },
        ],

        "beam_extra_param_sets": [
               # Candidate B measurement region
            {
                "beam_sigma_hit": 0.07,
                "beam_alpha_meas": 0.075,
                "beam_p_unknown": 0.10,
                "beam_p_out_of_map": 0.15,
                "beam_p_unexpected_known_free": 0.00,
                "beam_p_pred_below_min": 0.02,
                "beam_step": 2,
            },       
        ],
        
        # Motion model params
        "sigma_xy_motion": [0.12],                  # motion model uncertainty in x and y direction [m]
        "sigma_theta": [0.11],                      # motion model uncertainty in theta direction [rad]
        "ctrl_motion_fac": [0.1],                   # control motion factor for translational movement under uncertainty
        "ctrl_turn_fac": [0.15],                    # control turn factor for rotational movement under uncertainty
        
        # Proposal params (bound sets).
        # Each dict is one fixed combination of:
        # proposal_sigma_xy, proposal_sigma_theta, n_samples_dir
        # so these three values are sampled together (no Cartesian product among them).
        "proposal_param_sets": [
            {
                "proposal_sigma_xy": 0.06,      # Proposal window size in x/y direction [m]
                "proposal_sigma_theta": 0.025,   # proposal window size in theta direction [rad]
                "n_samples_dir": 3,             # samples per direction for proposal sampling (total samples = n_samples_dir^3)
            }
        ],
        # Proposal covariance scale/limit params (bound sets).
        # Each dict is one fixed combination propagated to sample_from_proposal_limit.

        "scale_limit_cov": [
            {
                "cov_std_scale": 0.5,
                "cov_max_std_xy": 1.0,
                "cov_max_std_theta": np.deg2rad(10),
                "min_std_xy": 0.0,
                "min_std_theta": np.deg2rad(0.0),
            },
            {
                "cov_std_scale": 0.6,
                "cov_max_std_xy": 1.0,
                "cov_max_std_theta": np.deg2rad(10),
                "min_std_xy": 0.0,
                "min_std_theta": np.deg2rad(0.0),
            },

        ],
        # TODO: Delete proposal values when no longer needed later on
        "proposal_alpha": [1.0],
        "proposal_beta": [1.0],

        # ScanMatcherParams (map extraction)
        "surface_radius_m": [0.2],      # TODO: Later change the name cause we search in a quadratic window not in circle
        "min_free_ratio": [0.4],

        # ICP jump thresholds
        "max_translation_jump": [0.7],
        "max_rotation_jump_deg": [45.0],
    }



def write_parameter_overview(path: str, n_repeats: int, override: bool = False) -> None:
    '''
    Write the experiment parameter overview to a JSON file for experiment reconstructability. 
    '''
    # dummy_pose = (0.0, 0.0, 0.0)
    dummy_pose = None
    file_exists = ResultWriter.create_path_and_check_if_file_exists(path=path)

    if file_exists and not override:
        print("\nParameter overview has not been saved because file already exists and override is set to False!")
        return

    axes = _grid_axes()
    example_experiment_params = next(generate_param_grid(start_pose=dummy_pose, n_repeats=1), None)

    payload = {
        # "used_meas_model": USED_MEAS_MODEL,
        "measurement_stddev": MEASUREMENT_STDDEV,
        "use_seed_list_for_measurement_noise": USE_SEED_LIST_FOR_MEASUREMENT_NOISE,
        "n_playback_steps": N_PLAYBACK_STEPS,
        "n_optimization_repeats": n_repeats,
        "seed_list": SEED_LIST,
        "start_pose": dummy_pose,
        "grid_axes": axes,
        "example_experiment_params": _to_jsonable(example_experiment_params) if example_experiment_params is not None else None,
    }

    with open(path, "w") as f:
        json.dump(_to_jsonable(payload), f, indent=2)

    print(f"\nParameter overview has been saved to:\n{path}")


def generate_param_grid(start_pose, n_repeats: int = 1):
    '''
    Defined the parameter grid for the RBPF SLAM optimization. This is a generator that yields ExperimentParams for
    each combination of parameters in the grid.
    '''
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    axes = _grid_axes()

    sigma_measurement = axes.get("sigma_measurement", [0.05])
    every_nth_beam_filter = axes.get("every_nth_beam_filter", [4])
    every_nth_beam_map = axes.get("every_nth_beam_map", [2])
    n_particles = axes.get("n_particles", [40])
    sigma_xy_motion = axes.get("sigma_xy_motion", [0.12])
    sigma_theta_motion = axes.get("sigma_theta", [0.05])
    ctrl_motion_fac = axes.get("ctrl_motion_fac", [0.1])
    ctrl_turn_fac = axes.get("ctrl_turn_fac", [0.15])
    neff_thres_ratio = axes.get("neff_thres_ratio", [None])
    if neff_thres_ratio is None:
        neff_thres_ratio = [None]
    proposal_param_sets = axes.get("proposal_param_sets", [])
    proposal_triplets = []
    for i, proposal_set in enumerate(proposal_param_sets):
        if not isinstance(proposal_set, dict):
            raise TypeError(
                f"proposal_param_sets[{i}] must be a dict, got {type(proposal_set)}"
            )

        try:
            proposal_triplets.append(
                (
                    float(proposal_set["proposal_sigma_xy"]),
                    float(proposal_set["proposal_sigma_theta"]),
                    int(proposal_set["n_samples_dir"]),
                )
            )
        except KeyError as exc:
            raise KeyError(
                f"proposal_param_sets[{i}] is missing required key: {exc}"
            ) from exc

    if not proposal_triplets:
        raise ValueError("No proposal parameter sets configured.")

    scale_limit_cov = axes.get("scale_limit_cov", [])
    scale_limit_cov_triplets = []
    for i, scale_limit_cov_set in enumerate(scale_limit_cov):
        if not isinstance(scale_limit_cov_set, dict):
            raise TypeError(
                f"scale_limit_cov[{i}] must be a dict, got {type(scale_limit_cov_set)}"
            )

        try:
            scale_limit_cov_triplets.append(
                (
                    float(scale_limit_cov_set["cov_std_scale"]),
                    float(scale_limit_cov_set["cov_max_std_xy"]),
                    float(scale_limit_cov_set["cov_max_std_theta"]),
                    float(scale_limit_cov_set["min_std_xy"]),
                    float(scale_limit_cov_set["min_std_theta"]),
                )
            )
        except KeyError as exc:
            raise KeyError(
                f"scale_limit_cov[{i}] is missing required key: {exc}"
            ) from exc

    if not scale_limit_cov_triplets:
        raise ValueError("No proposal covariance scale/limit sets configured.")

    meas_kernel_size = axes.get("meas_kernel_size", [1])
    beam_occ_thresh = axes.get("beam_occ_thresh", [1.4])
    beam_free_thresh = axes.get("beam_free_thresh", [0.8])
    beam_unknown_thresh = axes.get("beam_unknown_thresh", [0.5])
    beam_known_free_ratio_thresh = axes.get("beam_known_free_ratio_thresh", [0.4])

    beam_model_param_sets = axes.get("beam_model_param_sets", [])
    beam_extra_param_sets = axes.get("beam_extra_param_sets", [])
    if not beam_model_param_sets:
        raise ValueError("No beam model parameter sets configured.")
    if not beam_extra_param_sets:
        raise ValueError("No beam extra parameter sets configured.")

    for i, beam_model_set in enumerate(beam_model_param_sets):
        if not isinstance(beam_model_set, dict):
            raise TypeError(
                f"beam_model_param_sets[{i}] must be a dict, got {type(beam_model_set)}"
            )

    for i, beam_extra_set in enumerate(beam_extra_param_sets):
        if not isinstance(beam_extra_set, dict):
            raise TypeError(
                f"beam_extra_param_sets[{i}] must be a dict, got {type(beam_extra_set)}"
            )

    proposal_alpha = axes.get("proposal_alpha", [1.0])
    proposal_beta = axes.get("proposal_beta", [1.0])
    surface_radius_m = axes.get("surface_radius_m", [0.1])
    min_free_ratio = axes.get("min_free_ratio", [0.25])
    max_translation_jump = axes.get("max_translation_jump", [0.7])
    max_rotation_jump_deg = axes.get("max_rotation_jump_deg", [45.0])

    # Compute wheel separation
    wheel_separation = _compute_wheel_separation()

    for repeat_idx in range(1, n_repeats + 1):
        for (
            sigma_meas,
            every_nth_filter,
            every_nth_map,
            n_part,
            sigma_xy_m,
            sigma_theta_m,
            ctrl_motion,
            ctrl_turn,
            neff_thres,
            proposal_triplet,
            scale_limit_cov_triplet,
            kernel_size,
            beam_occ_th,
            beam_free_th,
            beam_unknown_ratio_th,
            beam_known_free_ratio_th,
            beam_model_set,
            beam_extra_set,
            alpha,
            beta,
            surface_r,
            min_free,
            max_trans_jump,
            max_rot_jump_deg,
        ) in itertools.product(
            sigma_measurement,
            every_nth_beam_filter,
            every_nth_beam_map,
            n_particles,
            sigma_xy_motion,
            sigma_theta_motion,
            ctrl_motion_fac,
            ctrl_turn_fac,
            neff_thres_ratio,
            proposal_triplets,
            scale_limit_cov_triplets,
            meas_kernel_size,
            beam_occ_thresh,
            beam_free_thresh,
            beam_unknown_thresh,
            beam_known_free_ratio_thresh,
            beam_model_param_sets,
            beam_extra_param_sets,
            proposal_alpha,
            proposal_beta,
            surface_radius_m,
            min_free_ratio,
            max_translation_jump,
            max_rotation_jump_deg,
        ):
            sigma_xy, sigma_theta, samples_dir = proposal_triplet
            cov_std_scale, cov_max_std_xy, cov_max_std_theta, min_std_xy, min_std_theta = scale_limit_cov_triplet
            neff_th = None if neff_thres is None else float(neff_thres) * n_part

            measurement_model_params = BeamRangeFinderMeasModelParams(
                occ_thresh=beam_occ_th,
                free_thresh=beam_free_th,
                unknown_ratio_thresh=beam_unknown_ratio_th,
                known_free_ratio_thresh=beam_known_free_ratio_th,

                sigma_hit=beam_extra_set["beam_sigma_hit"],
                w_hit=beam_model_set["beam_w_hit"],
                w_short=beam_model_set["beam_w_short"],
                lambda_short=beam_model_set["beam_lambda_short"],
                w_max=beam_model_set["beam_w_max"],
                w_rand=beam_model_set["beam_w_rand"],
                
                p_unknown=beam_extra_set["beam_p_unknown"],
                p_out_of_map=beam_extra_set["beam_p_out_of_map"],
                p_unexpected_known_free=beam_extra_set["beam_p_unexpected_known_free"],
                p_pred_below_min=beam_extra_set["beam_p_pred_below_min"],
                
                alpha_meas=beam_extra_set["beam_alpha_meas"],
                beam_step=beam_extra_set["beam_step"],
                eps=1e-12,
            )

            # Define experiment params for each run
            yield ExperimentParams(
                occupancy_params=OccupancyParams(
                    prior_probability=0.5,
                    min_distance_to_border=10.0,
                    increasing_probability=0.85,
                    decreasing_probability=0.15,
                    min_log_odds=-5.0,
                    max_log_odds=5.0,
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
                    max_n_points=1200,
                    downssample_grid_size=0.1,
                    max_correspondence_distance=0.4,
                    neighbors_pca=6,
                    max_iterations=5,
                    epsilon_rel=1e-3,
                    no_improvement_limit=3,
                    min_error=5e-4,
                    min_dtrans=1e-3, 
                    min_drot=1e-2,
                    min_points=20,
                    min_corresp=25,
                    min_hessian_rank=3,
                    max_hessian_condition=1e8,
                    max_translation_jump=max_trans_jump,
                    max_rotation_jump=np.deg2rad(max_rot_jump_deg),
                    max_acceptable_mean_error=0.15,
                ),
                robot_params=RobotParams(
                    wheel_separation=wheel_separation,
                ),
                scan_matcher_params=ScanMatcherParams(
                    occ_thres=1.4,
                    delta_r=0.6,
                    surface_radius_m=surface_r,
                    min_free_ratio=min_free,
                ),
                particle_params=ParticleParams(
                    n_particles=n_part,
                    start_pose=start_pose,
                ),
                motion_model_params=MotionModelParams(
                    sigma_x=sigma_xy_m,
                    sigma_y=sigma_xy_m,
                    sigma_theta=sigma_theta_m,
                    wheel_separation=wheel_separation,
                    ctrl_motion_fac=ctrl_motion,
                    ctrl_turn_fac=ctrl_turn,
                ),
                measurement_model_params=measurement_model_params,
                every_nth_scan_filter=every_nth_filter,
                every_nth_scan_map=every_nth_map,
                neff_threshold=neff_th,
                proposal_sigma_xy=sigma_xy,
                proposal_sigma_theta=sigma_theta,
                proposal_n_samples=samples_dir,
                cov_std_scale=cov_std_scale,
                cov_max_std_xy=cov_max_std_xy,
                cov_max_std_theta=cov_max_std_theta,
                min_std_xy=min_std_xy,
                min_std_theta=min_std_theta,
                meas_kernel_size=kernel_size,
                gaussian_sigma=0.05,
                proposal_alpha=alpha,
                proposal_beta=beta,
                measurement_noise_stddev=MEASUREMENT_STDDEV,
                used_meas_model=USED_MEAS_MODEL,
                tag=(
                    f"meas{sigma_meas}_nthf{every_nth_filter}_nmp{every_nth_map}_npart{n_part}_"
                    f"smxy{sigma_xy_m}_smth{sigma_theta_m}_cmf{ctrl_motion}_ctf{ctrl_turn}_"
                    f"neff{neff_th}_psig{sigma_xy}_psth{sigma_theta}_nsdir{samples_dir}_mks{kernel_size}_"
                    f"covss{cov_std_scale}_covmsxy{cov_max_std_xy}_covmsth{cov_max_std_theta}_"
                    f"minstdxy{min_std_xy}_minstdth{min_std_theta}_"
                    f"boct{beam_occ_th}_bfr{beam_free_th}_buth{beam_unknown_ratio_th}_bkfr{beam_known_free_ratio_th}_"
                    f"bsh{beam_extra_set['beam_sigma_hit']}_bwh{beam_model_set['beam_w_hit']}_"
                    f"bws{beam_model_set['beam_w_short']}_bls{beam_model_set['beam_lambda_short']}_"
                    f"bwm{beam_model_set['beam_w_max']}_bwr{beam_model_set['beam_w_rand']}_"
                    f"bpun{beam_extra_set['beam_p_unknown']}_bpoom{beam_extra_set['beam_p_out_of_map']}_"
                    f"bpukf{beam_extra_set['beam_p_unexpected_known_free']}_bppbm{beam_extra_set['beam_p_pred_below_min']}_"
                    f"bam{beam_extra_set['beam_alpha_meas']}_bs{beam_extra_set['beam_step']}_"
                    f"pa{alpha}_pb{beta}_surf{surface_r}_mfr{min_free}"
                ),
            )


def build_optimizer():
    # Init Playback runner
    factory = RBPFFactory()
    evaluator = RBPFEvaluator()
    playback_runner = PlaybackRunner(
        factory=factory,
        evaluator=evaluator,
        raw_odom_propagator=RawOdometryPropagator(),
    )

    # Init optimizer
    run_scorer = RunScorer()
    rbpf_optimizer = RBPFOptimizer(
        runner=playback_runner,
        scorer=run_scorer,
    )
    
    return rbpf_optimizer


def rbpf_tuning_pipeline():
    '''
    Normal sequential tuning pipeline that either runs the rbpf algorithm in single or multiprocessing variant.
    '''
    # Define vars
    ranked_run_list = []
    ranked_scored_path = OPTM_SUMMARY_PATH + "_" + "rank_scored.csv"
    agg_dataset_param_path = OPTM_SUMMARY_PATH + "_" + "agg_dataset_id_param.csv"
    agg_param_path = OPTM_SUMMARY_PATH + "_" + "agg_param.csv"
    ranked_param_overview_path = OPTM_SUMMARY_PATH + "_" + "ranked_param_overview.csv"

    # Init
    # Init playback loader and converter
    playback_loader = PlaybackLoader()
    playback_conv = PlaybackConverter()
    
    # Init optimizer
    rbpf_optimizer = build_optimizer()
    
    # Build result writer
    result_writer = ResultWriter()
    ranked_run_conv = RankedRunConverter()
    result_aggregator = ResultAggregator()
    step_processor = StepProcessor()

    # Store compact parameter overview (grid axes + one representative ExperimentParams)
    write_parameter_overview(
        path=PARAMETER_OVERVIEW_PATH,
        n_repeats=N_OPTIMIZATION_REPEATS,
        override=OVERRIDE_EXISTING_RESULTS,
    )

    # Load dataset and run tuning pipline
    # TODO: Put for loop into optimizer. Then do tqdm bar over all -> Full progress bar !
    optm_durations = []
    for playback_ds in PLAYBACK_DATA_LIST:
        # Load data
        print(f"\nLoading playback data:\nsuffix: {playback_ds.playback_suffix} \ndir: {playback_ds.playback_dir}")
        raw_playback_data = playback_loader.load(
            file_suffix=playback_ds.playback_suffix,
            filedir=playback_ds.playback_dir,
            n_steps=N_PLAYBACK_STEPS,
            ensure_start_pose=True,
            prompt_for_missing_start_pose=True,
        )

        start_pose = tuple(raw_playback_data.metadata["robot_start_pose"])
        print(f"Using start pose for tuning: {start_pose}")
    
        # Convert playback data
        playback_data = playback_conv.convert(
            raw_playback_data,
            measurement_stddev=None,
            min_range=MIN_SENSOR_RANGE,
            max_range=MAX_SENSOR_RANGE,
        )

        # Run optimizer
        ranked_runs, optm_duration_s = rbpf_optimizer.optimize(
            playback_data=playback_data,
            param_grid=generate_param_grid(start_pose=start_pose, n_repeats=N_OPTIMIZATION_REPEATS),
            seeds=SEED_LIST,
            dataset_id=playback_ds.playback_suffix,
            map_name=raw_playback_data.metadata.get("map", "unknown_map"),
            use_seed_list_for_measurement_noise=USE_SEED_LIST_FOR_MEASUREMENT_NOISE,
            keep_step_results=KEEP_STEP_RESULTS,
            run_storage_dir=RUN_STORAGE_DIR if STORE_MAP_DATA else None,
            store_map_data=STORE_MAP_DATA,
        )

        # Store ranked runs
        ranked_run_list.extend(ranked_runs)
        optm_durations.append(optm_duration_s)

    # Sort runs by score from lowest to highest
    ranked_run_list.sort(key=lambda ranked_run: ranked_run.score)
    
    # Clean optmization duration
    cleaned_optm_duratios = None
    cleaned_optm_duratios = [optm_dur_s for optm_dur_s in optm_durations if optm_dur_s is not None]
    if cleaned_optm_duratios is not None:
        overall_optm_duration_s = sum(cleaned_optm_duratios)
        print(f"\n\nFinished overall scan matching optimization in {overall_optm_duration_s} s")

    # Process step data and store into df
    if KEEP_STEP_RESULTS:
        step_trace_df = step_processor.process_ranked_runs(
            ranked_runs=ranked_run_list,
            pose_appendix=POSE_APPENDIX,
        )    

    # Aggregate results
    # Convert ranked runs to pandas DataFrame for easier analysis 
    ranked_run_df = ranked_run_conv.to_dataframe(ranked_run_list)

    # Rank results by score
    rank_scored_df = result_aggregator.rank_by_score(
        ranked_run_df=ranked_run_df,
        score_col="score",   
        ascending=True,
    )

    # Groupe and rank by playback data and seed
    agg_dataset_param_df = result_aggregator.aggregate_by_dataset_and_param(ranked_run_df)

    # Froupe and rank by paramters 
    agg_param_df = result_aggregator.aggregate_by_params(agg_dataset_param_df)

    # Build ranked parameter overview with one row per parameter_hash.
    ranked_param_overview_df = result_aggregator.build_ranked_parameter_overview(
        agg_param_df=agg_param_df,
        ranked_runs=ranked_run_list,
    )

    # Save results
    result_writer.write_dataframe_csv(
        path=ranked_scored_path,
        df=rank_scored_df,
        override=OVERRIDE_EXISTING_RESULTS,
        float_decimals=CSV_FLOAT_DECIMALS,
    )

    result_writer.write_dataframe_csv(
        path=agg_dataset_param_path,
        df=agg_dataset_param_df,
        override=OVERRIDE_EXISTING_RESULTS,
        float_decimals=CSV_FLOAT_DECIMALS,
    )

    result_writer.write_dataframe_csv(
        path=agg_param_path,
        df=agg_param_df,
        override=OVERRIDE_EXISTING_RESULTS,
        float_decimals=CSV_FLOAT_DECIMALS,
    )

    result_writer.write_dataframe_csv(
        path=ranked_param_overview_path,
        df=ranked_param_overview_df,
        override=OVERRIDE_EXISTING_RESULTS,
        float_decimals=CSV_FLOAT_DECIMALS,
    )

    # Save independent per-step diagnostic traces for each ranked run.
    # if KEEP_STEP_RESULTS:
    #     result_writer.write_run_steps_csv(
    #         output_path=STEP_TRACE_PATH,
    #         ranked_runs=ranked_run_list,
    #         override=OVERRIDE_EXISTING_RESULTS,
    #         float_decimals=CSV_FLOAT_DECIMALS,
    #     )
    # Save independent per-step diagnostic traces for each ranked run.
    if KEEP_STEP_RESULTS:
        result_writer.write_dataframe_csv(
            path=STEP_TRACE_PATH,
            df=step_trace_df,
            override=OVERRIDE_EXISTING_RESULTS,
            float_decimals=CSV_FLOAT_DECIMALS,
            cols_to_use=STEP_COLS_TO_USE,
            label="Step trace DataFrame",
        )

    print("\nTuning optimization completed.")


def rbpf_tuning_pipeline_multiprocessing():
    '''
    Tuning pipeline that trains the rbpf algorithm in parallel batches. The called optimizer creates n workers
    that process the parameter grid and corresponding seeds in parallel. The RBPF should notbe called in parallel
    here too, otherwise the multiprocessing will not work properly.
    '''
    # Define vars
    ranked_run_list = []
    ranked_scored_path = OPTM_SUMMARY_PATH + "_" + "rank_scored.csv"
    agg_dataset_param_path = OPTM_SUMMARY_PATH + "_" + "agg_dataset_id_param.csv"
    agg_param_path = OPTM_SUMMARY_PATH + "_" + "agg_param.csv"
    ranked_param_overview_path = OPTM_SUMMARY_PATH + "_" + "ranked_param_overview.csv"

    # Init
    # Init playback loader and converter
    playback_loader = PlaybackLoader()
    playback_conv = PlaybackConverter()
    
    # Init optimizer
    rbpf_optimizer = build_optimizer()
    
    # Build result writer
    result_writer = ResultWriter()
    ranked_run_conv = RankedRunConverter()
    result_aggregator = ResultAggregator()
    step_processor = StepProcessor()

    # Store compact parameter overview (grid axes + one representative ExperimentParams)
    write_parameter_overview(
        path=PARAMETER_OVERVIEW_PATH,
        n_repeats=N_OPTIMIZATION_REPEATS,
        override=OVERRIDE_EXISTING_RESULTS,
    )

    # Load dataset and run tuning pipline
    # TODO: Put for loop into optimizer. Then do tqdm bar over all -> Full progress bar !
    optm_durations = []
    for playback_ds in PLAYBACK_DATA_LIST:
        # Load data
        print(f"\nLoading playback data:\nsuffix: {playback_ds.playback_suffix} \ndir: {playback_ds.playback_dir}")
        raw_playback_data = playback_loader.load(
            file_suffix=playback_ds.playback_suffix,
            filedir=playback_ds.playback_dir,
            n_steps=N_PLAYBACK_STEPS,
            ensure_start_pose=True,
            prompt_for_missing_start_pose=True,
        )

        start_pose = tuple(raw_playback_data.metadata["robot_start_pose"])
        print(f"Using start pose for tuning: {start_pose}")
    
        # Convert playback data
        playback_data = playback_conv.convert(
            raw_playback_data,
            measurement_stddev=None,
            min_range=MIN_SENSOR_RANGE,
            max_range=MAX_SENSOR_RANGE,
        )

        # Run optimizer in parallel
        ranked_runs, optm_duration = rbpf_optimizer.optimize_parallel(
            playback_data=playback_data,
            param_grid=generate_param_grid(start_pose=start_pose, n_repeats=N_OPTIMIZATION_REPEATS),
            seeds=SEED_LIST,
            dataset_id=playback_ds.playback_suffix,
            map_name=raw_playback_data.metadata.get("map", "unknown_map"),
            use_seed_list_for_measurement_noise=USE_SEED_LIST_FOR_MEASUREMENT_NOISE,
            max_workers=NUMBER_OF_WORKERS,
            keep_step_results=KEEP_STEP_RESULTS,
            run_storage_dir=RUN_STORAGE_DIR if STORE_MAP_DATA else None,
            store_map_data=STORE_MAP_DATA,
        )

        # Store ranked runs
        ranked_run_list.extend(ranked_runs)
        optm_durations.append(optm_duration)


    # Sort runs by score from lowest to highest
    ranked_run_list.sort(key=lambda ranked_run: ranked_run.score)
    
    # Clean optmization duration
    cleaned_optm_duratios = None
    cleaned_optm_duratios = [optm_dur_s for optm_dur_s in optm_durations if optm_dur_s is not None]
    if cleaned_optm_duratios is not None:
        overall_optm_duration_s = sum(cleaned_optm_duratios)
        print(f"\n\nFinished overall scan matching optimization in {overall_optm_duration_s} s")

    # Process step data and store into df
    if KEEP_STEP_RESULTS:
        step_trace_df = step_processor.process_ranked_runs(
            ranked_runs=ranked_run_list,
            pose_appendix=POSE_APPENDIX,
        )    

    # Aggregate results
    # TODO: Adapt aggregate results to new tuning pipeline fo rbpf with multiple particles
    # Convert ranked runs to pandas DataFrame for easier analysis 
    ranked_run_df = ranked_run_conv.to_dataframe(ranked_run_list)

    # Rank results by score
    rank_scored_df = result_aggregator.rank_by_score(
        ranked_run_df=ranked_run_df,
        score_col="score",   
        ascending=True,
    )

    # # Groupe and rank by playback data and seed
    agg_dataset_param_df = result_aggregator.aggregate_by_dataset_and_param(ranked_run_df)

    # # Froupe and rank by paramters 
    agg_param_df = result_aggregator.aggregate_by_params(agg_dataset_param_df)

    # Build ranked parameter overview with one row per parameter_hash.
    ranked_param_overview_df = result_aggregator.build_ranked_parameter_overview(
        agg_param_df=agg_param_df,
        ranked_runs=ranked_run_list,
    )

    # # Save results
    result_writer.write_dataframe_csv(
        path=ranked_scored_path,
        df=rank_scored_df,
        override=OVERRIDE_EXISTING_RESULTS,
        float_decimals=CSV_FLOAT_DECIMALS,
        cols_to_exclude=SUMMARY_COLS_TO_EXCLUDE,
    )

    result_writer.write_dataframe_csv(
        path=agg_dataset_param_path,
        df=agg_dataset_param_df,
        override=OVERRIDE_EXISTING_RESULTS,
        float_decimals=CSV_FLOAT_DECIMALS,
    )

    result_writer.write_dataframe_csv(
        path=agg_param_path,
        df=agg_param_df,
        override=OVERRIDE_EXISTING_RESULTS,
        float_decimals=CSV_FLOAT_DECIMALS,
    )

    result_writer.write_dataframe_csv(
        path=ranked_param_overview_path,
        df=ranked_param_overview_df,
        override=OVERRIDE_EXISTING_RESULTS,
        float_decimals=CSV_FLOAT_DECIMALS,
    )


    # Save independent per-step diagnostic traces for each ranked run.
    if KEEP_STEP_RESULTS:
        result_writer.write_dataframe_csv(
            path=STEP_TRACE_PATH,
            df=step_trace_df,
            override=OVERRIDE_EXISTING_RESULTS,
            float_decimals=CSV_FLOAT_DECIMALS,
            cols_to_use=STEP_COLS_TO_USE,
            label="Step trace DataFrame",
        )

    print("\nTuning optimization completed.")



def main():
    # Attatch debugger
    # debugpy.listen(("localhost", 5678))
    # print("Waiting for debugger attach...")
    # debugpy.wait_for_client()

    # Initialize numba functions
    warmup_numba_functions()

    # RBPF tuning pipeline with RBPF step parallelization (multiprocessing)
    rbpf_tuning_pipeline()

    # RBPF tuning pipeline with pipeline parallelization (multiprocessing)
    # rbpf_tuning_pipeline_multiprocessing()
    
    


if __name__ == "__main__":
    main()
