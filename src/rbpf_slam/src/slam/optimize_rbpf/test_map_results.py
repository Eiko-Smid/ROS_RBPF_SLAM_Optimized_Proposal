#!/usr/bin/env python3

# import debugpy
# debugpy.listen(("localhost", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

from dataclasses import dataclass
import time
import numpy as np

from .playback_loader import load_playback_dataset
from ..scan_matcher.ogm_scan_matching import OGM 
from ..rbpf.scan_match_factory import OccupancyParams, SensorParams, MapParameter


PLAYBACK_DATA_PATH_PREF = '/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/scan_match/python_playback/1776425398_python_playback'
N_PLAYBACK_STEPS = None


@dataclass
class ExperimentParams:
    occupancy_params: OccupancyParams
    sensor_params: SensorParams
    map_param: MapParameter


def define_experiment_params():
    exp_param = ExperimentParams(
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
    )

    return exp_param


def init_ogm(exp_param: ExperimentParams) -> OGM:
    # init OGM algorithm
    ogm = OGM(
        map_parameter=exp_param.occupancy_params.min_distance_to_border,
        occupancy_parameter= [
            exp_param.occupancy_params.prior_probability,
            exp_param.occupancy_params.increasing_probability,
            exp_param.occupancy_params.decreasing_probability,
            exp_param.occupancy_params.min_log_odds,
            exp_param.occupancy_params.max_log_odds,
        ],
        sensor_parameter= [
            exp_param.sensor_params.min_sensor_range,
            exp_param.sensor_params.max_sensor_range,
        ]
    )


    # Init empty map with predefined prior probs
    ogm.init_map(
        map_width=exp_param.map_param.map_width,
        map_height=exp_param.map_param.map_height,
        grid_resolution=exp_param.map_param.grid_resolution_m
    )

    return ogm
    

def update_maps(ogm_classic: OGM, ogm_numba: OGM, steps):
    '''
    here we compare the old variant of update map with the new variant of update map (numba)
    '''
    for step in steps:
        # Extract measurements and true pose for the current step
        measurements = step.scan
        true_pose = step.true_pose

        # Extend maps
        extension_needed = True
        while(extension_needed):
            extension_needed = ogm_classic.map_extension_if_necessary(true_pose)
        
        extension_needed = True
        while(extension_needed):
            extension_needed = ogm_numba.map_extension_if_necessary(true_pose)
        
        # Update map based on pose and measurements
        ogm_classic.update_map_copy(
            measurements=measurements,
            pose=true_pose,
        )
        ogm_numba.update_map(
            measurements=measurements,
            pose=true_pose,
        )

    return ogm_classic, ogm_numba


def update_one_map(ogm: OGM, steps):
    '''
    here we compare the old variant of update map with the new variant of update map (numba)
    '''
    start_time = time.perf_counter()
    for i, step in enumerate(steps):
        # Extract measurements and true pose for the current step
        measurements = step.scan
        true_pose = step.true_pose

        # Extend maps
        extension_needed = True
        while(extension_needed):
            extension_needed = ogm.map_extension_if_necessary(true_pose)
        
        # Update map based on pose and measurements
        ogm.update_map(
            measurements=measurements,
            pose=true_pose,
        )

    end_time = time.perf_counter()
    total_time = (end_time - start_time) * 1000.0
    avg_time = total_time / i

    print(f"Update map total time: {total_time:.6f} ms")
    print(f"Update map avg time: {avg_time:.6f} ms")

    return ogm


def compare_maps(ogm_classic: OGM, ogm_numba: OGM):
    # Extract map data
    ogm_classic_map = ogm_classic.get_log_odds_map()
    ogm_numba_map = ogm_numba.get_log_odds_map()

    # Compare maps
    # Implement your comparison logic here
    diff = np.abs(ogm_classic_map - ogm_numba_map)
    print("max diff:", np.max(diff))
    print("num diff cells:", np.sum(diff > 1e-9))




def main():
    # Load playback data
    steps = load_playback_dataset(PLAYBACK_DATA_PATH_PREF, n_steps=N_PLAYBACK_STEPS)

    # Define parameters for the experiment
    exp_param = define_experiment_params()

    # Init OGM
    ogm_classic = init_ogm(exp_param)
    ogm_numba = init_ogm(exp_param)

    # Test OGM
    update_one_map(ogm_numba, steps)
    # update_maps(ogm_classic, ogm_numba, steps)
    # compare_maps(ogm_classic, ogm_numba)


if __name__ == "__main__":
    main()