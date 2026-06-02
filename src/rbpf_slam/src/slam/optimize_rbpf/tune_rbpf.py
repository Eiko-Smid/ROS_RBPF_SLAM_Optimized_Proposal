#!/usr/bin/env python3

# import debugpy
# debugpy.listen(("localhost", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

import itertools
import json
import numpy as np
from dataclasses import asdict, dataclass, is_dataclass

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
from .playback_runner import PlaybackRunner, RawOdometryPropagator
from .scorer import RunScorer
from .optimizer import RBPFOptimizer
from .result_writer import ResultWriter
from .aggregator import RankedRunConverter, ResultAggregator


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


18. Use deterministic sampling around scan match pose (Best run so far!!!)

    - Instead of taken random samples around scan matcher maxima we are using a deterministic sampling pattern. 
    - This test resulted in the best result so far. 
    - Unfortuntely the computation time for the proposal estimation increased a lot because we are now using 27 samples 
      instead of 10. 


19. Speedup of proposal computation

    - because we have more xjs now due to deterministic sampling, the proposal estimation time increased a lot.
    - To counter that we introduced a batch version for measurement likelihood and motion probability computation.
    - The vectorized computation enabled a much faster proposal estimation

    Results:
        proposal computation time before: proposal.estimate_proposal: 12.718832830819338 ms (count=20240)
        proposal estimation time after: proposal.estimate_proposal: 2.711988692958292 ms (count=20520)

        -> 4.7x speedup


    19_1: Speedversion test

        Result:

            - Same result as 18 only difference is the score because it depends on computation time and we are much faster now.
    
    
    19_2: Without clipping measurements
        
        - This time we run the measurement likelihood without clipping the distances to the nearest neighbor.
        - This should make the measurement likelihood more sensitive to bad correspondences but also more robust to good correspondences. 
        - Before we clipped the distances to 1.0 which means that we did not penalize bad correspondences more than a distance of 1.0 m.



20. Turtlebot map test     

    - Now we are testing our algorithm on the turtlebot map dataset. 
    - THis map has more unique structures than the cafe map 
    - Normally this should make it easier to localize and should lead to better results.

    Results:
        mean translation error: 0.3017 m -> worse than 0.058 (best cafe map result)
        mean rotation error: 0.6854 deg -> worse than 0.4139 deg (best cafe map result)

        - Before we tuned params on cafe map -> maybe overfitted to cafe map
        - Also we use deterministic sampling around scan match pose and use mu of proposal directly
        - Therefore all particles end up with the same weight and we have no probabilitic inside now 
        - We need to change this and therefore optimize the computation of the probabilits in the proposal estimation.


21. Added proposal metrics
    21.1: Proposal metrics test
        - We added some metrics to the proposal estimation to better understand the behavior of the proposal distribution 
          and its impact on the overall performance.



22. Analyze of turtle bot map based on new metrics 
    22.1. Turtle bot map run with fixed param

    22.1. Turtle bot map run with grid params


23. Added multiple metrics to analyze proposal

    23.1 Analyse which xj gets preferred and if it is correct

        - Short answer: 
            - The xj closest to the true pose is most of the time not the one with the highest weight. 
            - This means we can and must improve the weight compuation.

    
    23.2 Don't clip measurement model this time
            
        - We are not clipping the NN distances now in measurement lieklihood computation. 

        Results:
            - Made the pose errors of sm, proposal and xj worse
            - xj closest to true pose still doesn't get the highest weights most of the time
    
            
    23.3 Clip t0 1.5 insetad of 1 m

        Results:
            - Compared to clip of 1 m this also made the pose errorsr worse
            - Also the xj closest to true pose now got even slightly worse weights 
            
    
    23.4 Clip even clsoer (0.7 m)

        Results:
            - Kind of like same results than 23.3
            - 1m clipping seems to be near sweet spot for this dataset and param combination. 
            - Clipping is a double edged sword. It can make the measurement likelihood more
            robust to outliers but also less sensitive to good correspondences


24. Test different grid resolutions

    24.1 Higher gird resolution (0.06 m)
        
    Results:
        - The trans/rot error and pose errors are all higher than with grid resolution of 0.1 m

        
25. Analyse motion and measurement model probs

    -  Now we need to analyze why we follow the wrong xj most of the time

    25.1

        results:
            Motion model:
                - The motion model shows a normal correlation with the xj pose errors (0.342). 
                - However the model shows strong correlation with the weights (0.904)
                - This means the motion model is the main driver for the proposal distribution and its values are reasonable
                - From the "median_log_motion_range" we can also see that the distribution of the weights is quiet flat 0.381916
                  This values shows us how far away the max and min probs values of each step are away in average over an entire run

            Measurement model:
                - The measurement model shows a weaker correlation to the xj pose errors than the motion model (0.275)
                - However the model shows stronger correlation to the weights, but not as much as motion model (0.641)
                - This means that the measruement model don't benefit good xj poses, which is bad
                - Also the overall probs are to equal. They are big but they are not spreaded out well. We can see this from the 
                  "mean_log_meas_range" which has a value of 0.229785 which is even more flat than the motion model probs. 
                  Measurement models shouls have a high peak and therefore a high range between max and min values. 

            Conclusion:

                - For now the motion model is finde
                - But we need to make the measurement model a lot sharper
                - Currently we computing the mean over all distacnes between the map points and the beam endppoints. This makes
                  the measurement model too flat.


26. Adapt weight computaion

    26.1 Compute weights in log Odds space with scaling factors
        - We are now scaling the motion and measurement weights in klgo odds space
        
        Results:
            - We coldnt beat the ebst result so far
            - As expected the measurement weights are still as close to each other as before -> Flat measurement distribution
            - The correlations between the motion model and the xj pose err are as they were before. Lower alpha values reduce this fact
            - The corr between motion and weights could be reduced thanks to alpha
            
            - The corr between the measurment probs and the xjs errors have not really been increases. SO we still don't punish bad xjs
            - The corr between the meas probs and the weights could be increased. But this is meanigless if the models distribution is flat. 
        
    
    26.2 Use scaled measruement probs

        mean_error = np.mean(
            (distances / self.sigma) ** 2,
            axis=1,
        )

        k = 5.0
        scaled_mean = -0.5 * k * mean_error
        
        # probs = np.exp(-0.5 * mean_error)
        probs = np.exp(scaled_mean)

        
        26.2.1 K = 5    

            this does the following:
                old: 0.95 vs 0.90
                new: 0.95^5 vs 0.90^5
                    0.774 vs 0.590

            So it should make the measurements distribution more peaked

            
        26.2.1 K = 10
        

        26.2.3 k = 5 


27. Adapt measruement model

    - make changes accordign to chatgpt chat. Dont forget to negatze some of the former changes, 
    - But gpt wrote this already down.


    27.1 gmapping style likelihood computation on NN version

        - Here we used the old distances computation based on the trained map pints from the NN kdtree from scan matcher
        - But we replaced the likelihood computation with the gmapping style:
            - Define distance threshold
            - All distances above max_distance threshold are treated as invalid distances and are punished by the same value (no_hit)
            - 

        27.1.1 Old sampling window
            - Run with new likelihood computation but old sampling window (27 samples)

            Result:
                - For more see one note
                - More stable trans and rot errors but also worse than before (best 6 runs under 0.2 trans error)
                - Correlation between xj errors amd measurement probs is still very low.
                    -> Main issue stays the same


        27.1.2 new sampling window
            - Run with new likelihood computation but new sampling window (125 samples)

            Result:
                - See one drive
                - Increasing samples made results even worse
                - Main problem is still the measurement model itself. SO our NN approach alone is not enough


    27.2 gmapping style likelihood computation
        - We implemented the original gmapping measurement likelihood computation consisting of:
            - Estimate reflecting grid cell
            - Define a small grid search around the beam endpoint
            - Check if cell before occ candidate in beam direction is free -> valid candidate
            - Find closest distance among all valid candidates
            - Compute log_likelihood based on that
        
            27.2.1 Test run
        
                
    27.2 likelihood field model

        27.2.1 First run


        27.2.2 Make measurement likelihood more uncertain

          sigma_measurement = [0.2, 0.5, 1.0, 2.0]

    
    27.3 Beam range finder model 
    


28. After map shift problem solved

    - Before we had the followignn rpbolems
        ○ We only recorded data when the robot actually drives
		○ So the first playback stp was not at position (0, 0, 0)
		○ It was the position after we drove
		○ Then we also have no map values available
		○ In the secodn step we had occupied cells and perfomed scan matching
		○ But here we already had an offset in our map and our map didnt aligned with the real map from teh beginning on
		○ We carried that offset from tehbeginning on
		○ Because the inital created map was already in the wrong frame, our whole apporach failes

        • In the tutle bot map there was another error on top of that
        • We started directly with am translational error of 0.30331 because we spawned the robot at pose (0, 0.3, 0.0) but assumed the pose actually is (0.0, 0.0, 0.0)
        • Thats why the error was that big right away
        • At the end our drift almost exactly matched that value
        • This meaans we were never really off we simply had an offset from beginning on
    

    - Now we also tuned the scan mtacher on both maps with different seeds to find stable params. 
    - Here we will use these prams and test if we are now able to shift the proposal towards the xj with the min err
      to the true pose.

        28.1 turtle map

            28.1.1 First full run

                Results:

                    - Pose (trans, rot) errors almost identical to scan match only 
                    - Unfortunately the problem that the proposal doesn't follow the true pose is still there
                    - Since we are using the old measurement model we also don't shift the proposal towards the ebst xj
                      U can easily see this from the metrics:
                        - mean_log_motion_range = 0.428531
                        - mean_log_meas_range = 0.200502 -> measurement model more flat than motion model
                        - mean_corr_xjs_meas = 0.264045 -> Weak correlation between xj pose errr and measurement prob
                    
                TODOs:
                    - Update scorer to compare results over different runs and find best params for each map
                    - We need a new parameter search over both maps and different seeds 
                    - Then we need to compare the results and find best common params
                    - Then we need to try this for the different measurement models 
                    - If we are not able to improve we tzune icp for grid resolution of 0.05 instead of 0.1
                    - IF this still doesnt imrpvoe proposal than we need to change measurement model (ogm with mean positions stored)

        28.2 cafe map

            
            28.2.1 First full run

                - THis run failed completly
                - Doesn't make sense at all cause scan matcher made good estimate and then suddenly failed
                - The result was an increasing error in the pose which the system wasn't able to recover from 
                - That shoudnt have happened


            28.2.2 Test sm only

                - We overwrote the proposal results such that we got sm only results

                Result:
                    - Same results than in sm pipeline!
                    - That means proposal was the reason for teh bad results in 28.2.1 
                    - i will delete this cause its no longer needed and takes away memory 


29. Updated scorer
    - We updated the scorer to better reflect the new insights we got from the proposal analysis.

    29.1 Turtle bot map analysis with new scorer

        29.1.1 Full run with new scorer


        29.1.2 run with adapted scorer
    

    29.2 Cafe map analysis with new scorer


30. Create automated optimization pipeline
    - We created an automated pipelien that is able to run the rbpf on multiple maps, with multiple param sets on different seeds
      and finds the best overall performing metrics. 
    
      Attention!    
        Step duration will be longer here cause we iterate over different seeds!
    
        30.1 First run
            - cafe and turtle bot map
            - 72 different params
            - 3 seeds

            -> 2*72*3 = 864 runs in total


        30.2 Second run on new AWS indoor map 

'''


# Playback data path defs
OPTM_SUMMARY_PATH= '/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optimization_results/proposal_optm_30_2_summary'
STEP_TRACE_PATH = '/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optimization_results/proposal_optm_30_2_steps.csv'
PROPOSAL_WEIGHTS_PATH = '/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optimization_results/proposal_optm_30_2_proposal_weights.csv'
PARAMETER_OVERVIEW_PATH = '/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optimization_results/proposal_optm_30_2_params.json'

# OPTM_SUMMARY_PATH= '/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optimization_results/1779363559_test_summary'
# STEP_TRACE_PATH = '/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optimization_results/1779363559_test_steps.csv'
# PROPOSAL_WEIGHTS_PATH = '/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optimization_results/1779363559_test_proposal_weights.csv'
# PARAMETER_OVERVIEW_PATH = '/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optimization_results/1779363559_test_params.json'

USED_MEAS_MODEL = "Old_NN_Based"
# USED_MEAS_MODEL = "NN_Based_Gmap_Probs"
# USED_MEAS_MODEL = "GMAPPING"

CSV_FLOAT_DECIMALS = 6
OVERRIDE_EXISTING_RESULTS = False
N_PLAYBACK_STEPS = None             # Set an integer (e.g. 200) to use only the first N steps. None = all steps are used.
N_OPTIMIZATION_REPEATS = 1          # Number of full grid passes. 3 means each parameter combination is evaluated three times.
SEED_LIST = [22, 23, 56]
# SEED_LIST = [22, 56]

# Controls ONLY measurement-noise seeding behavior in optimizer:
# - True:  use values from SEED_LIST for deterministic per-seed measurement noise.
# - False: do not seed measurement noise (fresh random noise every run).
USE_SEED_LIST_FOR_MEASUREMENT_NOISE = True

# Define sttdev [m] to add noise to the playback measurements.
# Set to None to disable noise injection.
MEASUREMENT_STDDEV = 0.03
MIN_SENSOR_RANGE = 0.1
MAX_SENSOR_RANGE = 10.0 


# PLAYBACK_SUFFIX = "1779375646"        # Cafe map   

@dataclass
class PlaybackDataset:
    playback_dir: str
    playback_suffix: str


# Define datasets to load
# PLAYBACK_DATA_LIST = [
#     PlaybackDataset(
#         playback_dir="/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/python_playback/",
#         playback_suffix="1779363559",
#     ), PlaybackDataset(
#         playback_dir="/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/python_playback/",
#         playback_suffix="1779375646",
#     )
# ]

PLAYBACK_DATA_LIST = [
    PlaybackDataset(
        playback_dir="/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/python_playback/",
        playback_suffix="1780397517",
    )
]



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
        # General rbpf params
        "every_nth_beam_filter": [4],               # use every nth beam for proposal/scan matching
        "every_nth_beam_map": [2],                  # use every nth beam for map update
        "n_particles": [1],                         # number of particles in the RBPF
        "neff_threshold": [20],                     # Number of effective particles threshold for resampling

        # measurement model params
        "sigma_measurement": [0.06, 0.12, 0.15, 0.18],                # measurement uncertainty [m]
        "meas_kernel_size": [1],                    # Define search space size around beam endpoint for gmapping like measurement likelihood
        
        # Motion model params
        "sigma_xy_motion": [0.08, 0.12, 0.2],       # motion model uncertainty in x and y direction [m]
        "sigma_theta": [0.03, 0.07, 0.1],           # motion model uncertainty in theta direction [rad]
        "ctrl_motion_fac": [0.1],                   # control motion factor for translational movement under uncertainty
        "ctrl_turn_fac": [0.15],                    # control turn factor for rotational movement under uncertainty
        
        # Proposal params (bound sets).
        # Each dict is one fixed combination of:
        # proposal_sigma_xy, proposal_sigma_theta, n_samples_dir
        # so these three values are sampled together (no Cartesian product among them).
        "proposal_param_sets": [
            {
                "proposal_sigma_xy": 0.05,      # # Proposal window size in x/y direction [m]
                "proposal_sigma_theta": 0.02,   # proposal window size in theta direction [rad]
                "n_samples_dir": 3,             # samples per direction for proposal sampling (total samples = n_samples_dir^3)
            },
            {
                "proposal_sigma_xy": 0.1,
                "proposal_sigma_theta": 0.06,
                "n_samples_dir": 5,
            },
        ],
        # TODO: Delete proposal values when no longer needed later on
        "proposal_alpha": [1.0],
        "proposal_beta": [1.0],

        # ScanMatcherParams (map extraction)
        "surface_radius_m": [0.2],      # TODO: Later change the name cause we search in a quadratic window not in circle
        "min_free_ratio": [0.25],
    }


# def _grid_axes() -> dict:
#     return {
#         # General rbpf params
#         "every_nth_beam_filter": [4],               # use every nth beam for proposal/scan matching
#         "every_nth_beam_map": [2],                  # use every nth beam for map update
#         "n_particles": [1],                         # number of particles in the RBPF
#         "neff_threshold": [20],                     # Number of effective particles threshold for resampling

#         # measurement model params
#         "sigma_measurement": [0.06],                # measurement uncertainty [m]
#         "meas_kernel_size": [1],                    # Define search space size around beam endpoint for gmapping like measurement likelihood
        
#         # Motion model params
#         "sigma_xy_motion": [0.2],       # motion model uncertainty in x and y direction [m]
#         "sigma_theta": [0.07],           # motion model uncertainty in theta direction [rad]
#         "ctrl_motion_fac": [0.1],                   # control motion factor for translational movement under uncertainty
#         "ctrl_turn_fac": [0.15],                    # control turn factor for rotational movement under uncertainty
        
#         # Proposal params (bound sets).
#         # Each dict is one fixed combination of:
#         # proposal_sigma_xy, proposal_sigma_theta, n_samples_dir
#         # so these three values are sampled together (no Cartesian product among them).
#         "proposal_param_sets": [
#             {
#                 "proposal_sigma_xy": 0.05,      # # Proposal window size in x/y direction [m]
#                 "proposal_sigma_theta": 0.02,   # proposal window size in theta direction [rad]
#                 "n_samples_dir": 3,             # samples per direction for proposal sampling (total samples = n_samples_dir^3)
#             },
#             {
#                 "proposal_sigma_xy": 0.1,
#                 "proposal_sigma_theta": 0.06,
#                 "n_samples_dir": 5,
#             },
#         ],
#         # TODO: Delete proposal values when no longer needed later on
#         "proposal_alpha": [1.0],
#         "proposal_beta": [1.0],

#         # ScanMatcherParams (map extraction)
#         "surface_radius_m": [0.2],      # TODO: Later change the name cause we search in a quadratic window not in circle
#         "min_free_ratio": [0.25],
#     }


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
    example_params = next(generate_param_grid(start_pose=dummy_pose, n_repeats=1), None)

    payload = {
        "used_meas_model": USED_MEAS_MODEL,
        "measurement_stddev": MEASUREMENT_STDDEV,
        "use_seed_list_for_measurement_noise": USE_SEED_LIST_FOR_MEASUREMENT_NOISE,
        "n_playback_steps": N_PLAYBACK_STEPS,
        "n_optimization_repeats": n_repeats,
        "seed_list": SEED_LIST,
        "start_pose": dummy_pose,
        "grid_axes": axes,
        "example_experiment_params": _to_jsonable(example_params) if example_params is not None else None,
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
    neff_threshold = axes.get("neff_threshold", [20])
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

    meas_kernel_size = axes.get("meas_kernel_size", [1])
    proposal_alpha = axes.get("proposal_alpha", [1.0])
    proposal_beta = axes.get("proposal_beta", [1.0])
    surface_radius_m = axes.get("surface_radius_m", [0.1])
    min_free_ratio = axes.get("min_free_ratio", [0.25])

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
            proposal_triplet,
            kernel_size,
            alpha,
            beta,
            surface_r,
            min_free,
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
            proposal_triplets,
            meas_kernel_size,
            proposal_alpha,
            proposal_beta,
            surface_radius_m,
            min_free_ratio,
        ):
            sigma_xy, sigma_theta, samples_dir = proposal_triplet

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
                    max_translation_jump=0.3,
                    max_rotation_jump=np.deg2rad(45.0),
                    max_acceptable_mean_error=0.15,
                ),
                robot_params=RobotParams(
                    wheel_separation=wheel_separation,
                ),
                scan_matcher_params=ScanMatcherParams(
                    occ_thres=1.2,
                    delta_r=0.4,
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
                measurement_model_params=MeasurementModelParams(
                    sigma_measurement=sigma_meas,
                ),
                every_nth_scan_filter=every_nth_filter,
                every_nth_scan_map=every_nth_map,
                neff_threshold=neff_th,
                proposal_sigma_xy=sigma_xy,
                proposal_sigma_theta=sigma_theta,
                proposal_n_samples=samples_dir,
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
                    f"pa{alpha}_pb{beta}_surf{surface_r}_mfr{min_free}"
                ),
            )


def build_optimizer():
    # Init Playback runner
    scan_match_fac = RBPFFactory()
    scan_match_eval = RBPFEvaluator()
    scan_match_playback_run = PlaybackRunner(
        factory=scan_match_fac,
        evaluator=scan_match_eval,
        raw_odom_propagator=RawOdometryPropagator(),
    )

    # Init optimizer
    run_scorer = RunScorer()
    rbpf_optimizer = RBPFOptimizer(
        runner=scan_match_playback_run,
        scorer=run_scorer,
    )
    
    return rbpf_optimizer



def main():
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

    # Store compact parameter overview (grid axes + one representative ExperimentParams)
    write_parameter_overview(
        path=PARAMETER_OVERVIEW_PATH,
        n_repeats=N_OPTIMIZATION_REPEATS,
        override=OVERRIDE_EXISTING_RESULTS,
    )

    # Load dataset and run tuning pipline
    # TODO: Put for loop into optimizer. Then do tqdm bar over all -> Full progress bar !
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
        ranked_runs = rbpf_optimizer.optimize(
            playback_data=playback_data,
            param_grid=generate_param_grid(start_pose=start_pose, n_repeats=N_OPTIMIZATION_REPEATS),
            seeds=SEED_LIST,
            dataset_id=playback_ds.playback_suffix,
            map_name=raw_playback_data.metadata.get("map", "unknown_map"),
            use_seed_list_for_measurement_noise=USE_SEED_LIST_FOR_MEASUREMENT_NOISE,
        )

        # Store ranked runs
        ranked_run_list.extend(ranked_runs)

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
    # result_writer.write_run_steps_csv(
    #     output_path=STEP_TRACE_PATH,
    #     ranked_runs=ranked_run_list,
    #     override=OVERRIDE_EXISTING_RESULTS,
    #     float_decimals=CSV_FLOAT_DECIMALS,
    # )

    # Save per-step, per-proposal-sample diagnostics (raw weights/motion/meas).
    # TODO: Add proposal weights again
    # result_writer.write_proposal_weights_csv(
    #     output_path=PROPOSAL_WEIGHTS_PATH,
    #     ranked_runs=ranked_run_list,
    #     override=OVERRIDE_EXISTING_RESULTS,
    #     float_decimals=CSV_FLOAT_DECIMALS,
    # )

    print("\nTuning optimization completed.")
    


if __name__ == "__main__":
    main()