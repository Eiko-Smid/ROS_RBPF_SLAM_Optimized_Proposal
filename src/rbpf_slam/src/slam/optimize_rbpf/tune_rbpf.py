#!/usr/bin/env python3

# import debugpy
# debugpy.listen(("localhost", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

import itertools
import json
import numpy as np
from dataclasses import asdict, is_dataclass

from .playback_defs import ExperimentParams, PlaybackData
# from .playback_loader import load_playback_dataset

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

9.6 Added possibility to run the same grid param several time in a row. This is to check the stability of the results.
- We ran the same grid parameters 5 times. 
- We used the same playback data and the same code in each run.
- Unfortunately we ended up with totally different results
- We must check if numba variant produces the same results than old ogm. IF so it's not the fault of the new optimized code
- If not the numba version is wrong


10: Implemented seed

- Made it possible to create determinitic runs by setting a global seed.


11. used new dataset 

- We are still using the cafe map here but another dataset is used. 


12: Updated ICP algorithm

- Before we used the tf of the icp no matter if it succeeded or not. 
- This could lead to problems if the icp failed and returned a bad tf.
    Bad tf -> bad pose for propüosal estimation
- We added some safety checks and added an inidcator wheather to use or not use the returned transformation.

    12.1 Full run

        - We ended up with a large error in transltion. About 0.1 m more than before icp update
        - But thats definitely because the icp tfs are often declared as not valid.
        - 

    12.2 ICP param change

        - We are changing the params as follows:
            max_translation_jump=0.8,  # was 0.3
            max_rotation_jump=np.deg2rad(120.0),  # was 60
            max_acceptable_mean_error=0.15 # was 2.5e-3 = 0.0025


    12.3 



14. New icp transformation update

    - We are now using: T = dT @ T insetad of T = T + dT
    - The one before is mathematically only valid for small dT vals.

    14.1 With new transformation update


    14.2 Same params old TF update


15. Test with scan matcher pose insetad of proposal pose

    15.1 Full run with scan matcher pose and proposal weights
        - Low uncertainty values
        -> worse than scan matching only variant

    15.2 Full run with scan matcher pose and proposal weights
        - High uncertainty values
        -> made it worse

    15.3 No uncertainty in scan match fallback
        - Before everytime sm failed we added noise to odom and prdeict the pose based on noisy odom
        - Now in fallback we used raw odom without adding noise to predict particle pose
        -> Result is exactly as good as sm only variant

        
16. Test rbpf with proposal pose but no uncertainty in scan match fallback
    - Better result than 15.1
    - But still worse than scan matching only variant.


17. use mean of proposal instead of sampling a value

    17.1 Full run by adapting measurement and motion model uncertainty params.

        "sigma_measurement": [0.05, 0.15],
        "every_nth_beam_filter": [4],
        "every_nth_beam_map": [2],
        "n_particles": [40],
        "sigma_xy_motion": [0.08, 0.18],
        "sigma_theta": [0.05, 0.1],
        "ctrl_motion_fac": [0.1],
        "ctrl_turn_fac": [0.15],
        "neff_threshold": [20],
        "proposal_sigma_xy": [0.05],
        "proposal_sigma_theta": [0.02],
        "proposal_n_samples": [10],

    17.2 Use best motion and uncertainty from 17.1 and adapt proposal params only (TODO)


18. Use deterministic sampling around scan match pose


19. Speedup

    - because we have more xjs nowe in deterministic sampling, the proposal estimation time increased a lot.
    - To counter that we introduced a batch version for measurement lieklihood and motion probability computation.

    Results:
        proposal compuation time before: proposal.estimate_proposal: 12.718832830819338 ms (count=20240)
        proposal estimation time after: proposal.estimate_proposal: 2.711988692958292 ms (count=20520)

        -> 4.7x speedup
        
'''


# Playback data path defs
# OPTIMIZATION_RESULT_PATH= '/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optimization_results/1777891056_optm_17_1_summary.csv'
# STEP_TRACE_PATH = '/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optimization_results/1777891056_optm_17_1_steps.csv'
# PARAMETER_OVERVIEW_PATH = '/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optimization_results/1777891056_optm_17_1_params.json'


OPTM_SUMMARY_PATH= '/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optimization_results/1777891056_optm_19_1_summary.csv'
STEP_TRACE_PATH = '/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optimization_results/1777891056_optm_19_1_steps.csv'
PARAMETER_OVERVIEW_PATH = '/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optimization_results/1777891056_optm_19_1_params.json'


CSV_FLOAT_DECIMALS = 4
OVERRIDE_EXISTING_RESULTS = False
N_PLAYBACK_STEPS = None             # Set an integer (e.g. 200) to use only the first N steps. None = all steps are used.
N_OPTIMIZATION_REPEATS = 1          # Number of full grid passes. 3 means each parameter combination is evaluated three times.
# SEED_LIST = [22, 23, 24, 56]
SEED_LIST = [22]

PLAYBACK_DIR = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/python_playback/"
PLAYBACK_SUFFIX = "1777891056"


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


def _grid_axes() -> dict:
    return {
        # "sigma_measurement": [0.08, 0.15],
        # "every_nth_beam_filter": [4],
        # "every_nth_beam_map": [2],
        # "n_particles": [40],
        # "sigma_xy_motion": [0.12, 0.2],
        # "sigma_theta": [0.05],
        # "ctrl_motion_fac": [0.1],
        # "ctrl_turn_fac": [0.15],
        # "neff_threshold": [20],
        # "proposal_sigma_xy": [0.05],
        # "proposal_sigma_theta": [0.02],
        # "proposal_n_samples": [10],

        "sigma_measurement": [0.15],
        "every_nth_beam_filter": [4],
        "every_nth_beam_map": [2],
        "n_particles": [40],
        "sigma_xy_motion": [0.12],
        "sigma_theta": [0.05],
        "ctrl_motion_fac": [0.1],
        "ctrl_turn_fac": [0.15],
        "neff_threshold": [20],
        "proposal_sigma_xy": [0.05],
        "proposal_sigma_theta": [0.02],
        "proposal_n_samples": [10],
    }


def write_parameter_overview(path: str, n_repeats: int, override: bool = False) -> None:
    file_exists = ResultWriter.create_path_and_check_if_file_exists(path=path)

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


def generate_param_grid(n_repeats: int = 1):
    '''
    Defined the parameter grid for the RBPF SLAM optimization. This is a generator that yields ExperimentParams for
    each combination of parameters in the grid.
    '''
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    axes = _grid_axes()

    sigma_measurement = axes["sigma_measurement"]
    every_nth_beam_filter = axes["every_nth_beam_filter"]
    every_nth_beam_map = axes["every_nth_beam_map"]
    n_particles = axes["n_particles"]
    sigma_xy_motion = axes["sigma_xy_motion"]
    sigma_theta_motion = axes["sigma_theta"]
    ctrl_motion_fac = axes["ctrl_motion_fac"]
    ctrl_turn_fac = axes["ctrl_turn_fac"]
    neff_threshold = axes["neff_threshold"]
    proposal_sigma_xy = axes["proposal_sigma_xy"]
    proposal_sigma_theta = axes["proposal_sigma_theta"]
    proposal_n_samples = axes["proposal_n_samples"]

    # OGM param
    # TODO Add ogm param later

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
            neff_th,
            sigma_xy,
            sigma_theta,
            n_samples,
        ) in itertools.product(
            sigma_measurement,
            every_nth_beam_filter,
            every_nth_beam_map,
            n_particles,
            sigma_xy_motion,
            sigma_theta_motion,
            ctrl_motion_fac,
            ctrl_turn_fac,
            neff_threshold,
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
                    max_correspondence_distance=0.6,
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
                    max_translation_jump=0.8,
                    max_rotation_jump=np.deg2rad(120.0),
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
                motion_model_params=MotionModelParams(
                    sigma_x=sigma_xy_m,
                    sigma_y=sigma_xy_m,
                    sigma_theta=sigma_theta_m,
                    wheel_separation=wheel_separation,
                    ctrl_motion_fac=ctrl_motion,
                    ctrl_turn_fac=ctrl_turn,
                ),
                measurement_model_params=MeasurementModelParams(
                    sigma_measurement=sigma_meas,
                ),
                every_nth_scan_filter=every_nth_filter,
                every_nth_scan_map=every_nth_map,
                neff_threshold=neff_th,
                proposal_sigma_xy=sigma_xy,
                proposal_sigma_theta=sigma_theta,
                proposal_n_samples=n_samples,
                tag=(
                    f"meas{sigma_meas}_nthf{every_nth_filter}_nmp{every_nth_map}_npart{n_part}_"
                    f"smxy{sigma_xy_m}_smth{sigma_theta_m}_cmf{ctrl_motion}_ctf{ctrl_turn}_"
                    f"neff{neff_th}_psig{sigma_xy}_psth{sigma_theta}_pns{n_samples}_rep{repeat_idx}"
                ),
            )



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
    playback_loader = PlaybackLoader()
    raw_playback_data = playback_loader.load(
        file_suffix=PLAYBACK_SUFFIX,
        filedir=PLAYBACK_DIR,
        n_steps=N_PLAYBACK_STEPS,
    )

    # Convert playback data
    playback_conv = PlaybackConverter()
    playback_data = playback_conv.convert(raw_playback_data)

    # Init optimizer
    scan_match_optimizer = build_optimizer()

    # Build result writer
    result_writer = ResultWriter()

    # Store compact parameter overview (grid axes + one representative ExperimentParams)
    write_parameter_overview(
        path=PARAMETER_OVERVIEW_PATH,
        n_repeats=N_OPTIMIZATION_REPEATS,
        override=OVERRIDE_EXISTING_RESULTS,
    )

    # Run optimizer
    ranked_runs = scan_match_optimizer.optimize(
        playback_data=playback_data,
        param_grid=generate_param_grid(n_repeats=N_OPTIMIZATION_REPEATS),
        seeds=SEED_LIST,
    )


    # Run optimizer without proposal pose (scan matcher pose is used instead)
    # ranked_runs = scan_match_optimizer.optimize_without_proposal_pose(
    #     playback_data=playback_data,
    #     param_grid=generate_param_grid(n_repeats=N_OPTIMIZATION_REPEATS),
    #     seeds=SEED_LIST,
    # )


    # Save results
    result_writer.write_ranked_runs_csv(
        path=OPTM_SUMMARY_PATH,
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