#!/usr/bin/env python3

# import debugpy
# debugpy.listen(("localhost", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

import itertools

from .playback_defs import ExperimentParams, PlaybackData
from .playback_loader import load_playback_dataset
from ..rbpf.rbpf import RBPFFactory, ParticleParams, MotionModelParams, MeasurementModelParams
from ..rbpf.scan_match_factory import (
    OccupancyParams,
    SensorParams,
    MapParameter,
    ICPParams,
    RobotParams,
    ScanMatcherParams,
    ScanMatchFactory
)

from .evaluator import RBPFEvaluator
from .playback_runner import PlaybackRunner
from .scorer import RunScorer
from .optimizer import ScanMatcherOptimizer
from .result_writer import ResultWriter


'''
9.0 Run after numba for map update
- The mean_tran_err was at 7.42 m

9.1 Another run with numba for map update
- Here we were at 4.83

9.2 Run without numba but used new method which already had (35 % speedup)
- Already way closer to the original results
- Here we had mean_tran_err = 1.2 
- But i am still unsure why the results differ that much

9.3 with completly old ogm (despite angle normalization)
- mean trasn err = 0.418

9.4 with completly old ogm (despite angle normalization)
- mean trasn err = 0.535

9.5 With corrected numba version
- mean trans err = 

9.6 Added possibility to run the dsame grid several time in a row. This is to check the stability of the results.
- We ran the same grid parameters 5 times. 
- We used the same playback data and the same code in each run.
- Unfortunately we ended up with totally different results
- We must check if numba variant produces the same results than old ogm. IF so it's not the fault of the new optimized code
- If not the numba version is wrong

'''


# Playback data path defs
# PLAYBACK_DATA_PATH_PREF = '/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/scan_match/python_playback/test_python_playback'
# OPTIMIZATION_RESULT_PATH= '/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/scan_match/optimization_results/test_optm.csv'
PLAYBACK_DATA_PATH_PREF = '/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/scan_match/python_playback/1776425398_python_playback'
OPTIMIZATION_RESULT_PATH= '/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/scan_match/optimization_results/1776425398_optm_9_6_map_speedup.csv'
STEP_TRACE_PATH = '/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/scan_match/optimization_results/1776425398_optm_9_6_map_speedup_steps.csv'
CSV_FLOAT_DECIMALS = 4
OVERRIDE_EXISTING_RESULTS = False
N_PLAYBACK_STEPS = None     # Set an integer (e.g. 200) to use only the first N steps. None = all steps are used.
N_OPTIMIZATION_REPEATS = 5  # Number of full grid passes. 3 means each parameter combination is evaluated three times.


def generate_param_grid(n_repeats: int = 1):
    '''
    Defined the parameter grid for the RBPF SLAM optimization. This is a generator that yields ExperimentParams for
    each combination of parameters in the grid.
    '''
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    # Motion model params
    # sigma_x = [0.05, 0.1, 0.2]
    # sigma_y = [0.05, 0.1, 0.2]
    # sigma_theta = [0.05, 0.1, 0.2]
    # ctrl_motion_fac
    # ctrl_turn_fac

    # Measurement parameter
    sigma_measurement = [0.2]
    # sigma_measurement = [0.1, 0.2, 0.4]
    # every_nth_beam = [5, 10, 20]
    every_nth_beam = [1]
    
    # RBPF param
    # n_particles = [30, 40, 50]
    n_particles = [40]

    # Proposal parameter
    # proposal_sigma_xy = [0.1, 0.2]
    proposal_sigma_xy = [0.1]
    proposal_sigma_theta = [0.05]
    proposal_n_samples = [10]

    # OGM param
    # TODO Add ogm param later

    # Compute wheel separation
    h_chassis= 0.15
    dist_chassis_to_ground= h_chassis/5
    r_wheel= h_chassis/2 + dist_chassis_to_ground
    w_wheel= 0.3 * r_wheel
    r_chassis= 0.25
    wheel_separation= 2 * r_chassis + w_wheel


    for repeat_idx in range(1, n_repeats + 1):
        for sigma_meas, every_nth, n_part, sigma_xy, sigma_theta, n_samples in itertools.product(
            sigma_measurement,
            every_nth_beam,
            n_particles,
            proposal_sigma_xy,
            proposal_sigma_theta,
            proposal_n_samples,
        ):
            # Define experiment params for each run
            yield ExperimentParams(
                occupancy_params=OccupancyParams(
                    prior_probability=0.5,
                    min_distance_to_border=10.0,
                    increasing_probability=0.7,
                    decreasing_probability=0.35,
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
                    grid_resolution_m=0.05,
                ),
                icp_params=ICPParams(
                    max_n_points=400,
                    max_correspondence_distance=0.6,
                    neighbors_pca=10,
                    max_iterations=5,
                    epsilon_rel=1e-3,
                    no_improvement_limit=3,
                    min_error=5e-4,
                    min_dtrans=1e-3, 
                    min_drot=1e-2,
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
                motion_model_params=MotionModelParams(
                    sigma_x=0.2,
                    sigma_y=0.2, 
                    sigma_theta=0.15, 
                    wheel_separation=wheel_separation,
                    ctrl_motion_fac=0.1,
                    ctrl_turn_fac=0.20, 
                ),
                measurement_model_params=MeasurementModelParams(
                    sigma_measurement=sigma_meas,
                ),
                every_nth_scan=every_nth,
                proposal_sigma_xy=sigma_xy,
                proposal_sigma_theta=sigma_theta,
                proposal_n_samples=n_samples,
                tag=(
                    f"meas{sigma_meas}_nth{every_nth}_npart{n_part}_"
                    f"psig{sigma_xy}_psth{sigma_theta}_pns{n_samples}_rep{repeat_idx}"
                ),
            )


# def build_optimizer():
#     # Init objects
#     # Init Playback runner
#     rbpf_fac = ScanMatcherFactory()
#     scan_match_eval = ScanMatcherEvaluator()
#     scan_match_playback_run = PlaybackRunner(
#         factory=rbpf_fac,
#         evaluator=scan_match_eval,
#     )

#     # Init optimizer
#     run_scorer = RunScorer()
#     scan_match_optimizer = ScanMatcherOptimizer(
#         runner=scan_match_playback_run,
#         scorer=run_scorer,
#     )
    
#     return scan_match_optimizer



def build_optimizer():
    # Init Playback runner
    scan_match_fac = RBPFFactory()
    scan_match_eval = RBPFEvaluator()
    scan_match_playback_run = PlaybackRunner(
        factory=scan_match_fac,
        evaluator=scan_match_eval,
    )

    # Init optimizer
    run_scorer = RunScorer()
    scan_match_optimizer = ScanMatcherOptimizer(
        runner=scan_match_playback_run,
        scorer=run_scorer,
    )
    
    return scan_match_optimizer



def main():
    # Load playback data
    steps = load_playback_dataset(
        base_path_prefix=PLAYBACK_DATA_PATH_PREF,
        n_steps=N_PLAYBACK_STEPS,
    )

    # Build playback data
    playback_data = PlaybackData(
        step_data_list=steps,
    )

    # Init optimizer
    scan_match_optimizer = build_optimizer()

    # Build result writer
    result_writer = ResultWriter()

    # Run optimizer
    ranked_runs = scan_match_optimizer.optimize(
        playback_data=playback_data,
        param_grid=generate_param_grid(n_repeats=N_OPTIMIZATION_REPEATS),
    )

    # Save results
    result_writer.write_ranked_runs_csv(
        path=OPTIMIZATION_RESULT_PATH,
        ranked_runs=ranked_runs,
        override=OVERRIDE_EXISTING_RESULTS,
        float_decimals=CSV_FLOAT_DECIMALS,
    )

    # Save independent per-step diagnostic traces for each ranked run.
    result_writer.write_ranked_step_traces_csv(
        output_path=STEP_TRACE_PATH,
        ranked_runs=ranked_runs,
        override=OVERRIDE_EXISTING_RESULTS,
        float_decimals=CSV_FLOAT_DECIMALS,
    )

    print("Test success")
    


if __name__ == "__main__":
    main()