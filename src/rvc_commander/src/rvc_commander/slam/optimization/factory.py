#!/usr/bin/env python3

from scan_match_playback_def import (
    PlaybackData,
    ExperimentParams,
)

from icp_scan_matching import IterativeClosestPoint
from ogm_scan_matching import OGM
from scan_matcher import ScanMatcher


class ScanMatcherFactory:
    '''
    Factory class for creating ScanMatcher instances.
    '''
    @staticmethod
    def build(playback_data: PlaybackData, params: ExperimentParams):
        '''
        Gets the playback data and experiment parameters, builds a ScanMatcher instance and returns it.
        '''
        # Extract map data from playback data
        map_data = playback_data.map_data

        # init OGM algorithm
        ogm = OGM(
            map_parameter=map_data.min_distance_to_border,
            occupancy_parameter=[
                map_data.occupancy_param.prior_probability,
                map_data.occupancy_param.increasing_probability,
                map_data.occupancy_param.decreasing_probability,
                map_data.occupancy_param.min_log_odds,
                map_data.occupancy_param.max_log_odds,
            ],
            sensor_parameter=[
                map_data.sensor_param.min_sensor_range,
                map_data.sensor_param.max_sensor_range,
            ],
        )

        # Init the map
        ogm.init_map_from_map(
            log_odds_map=map_data.log_odds_map,
            grid_resolution=map_data.grid_resolution_m
        )

        # Init ICP algorithm
        icp = IterativeClosestPoint(
            stop_params={
                "max_iterations": params.icp.max_iterations,
                "epsilon_rel": params.icp.epsilon_rel,
                "no_improvement_limit": params.icp.no_improvement_limit,
                "min_error": params.icp.min_error,
                "min_dtrans": params.icp.min_dtrans,
                "min_drot": params.icp.min_drot,
            },
            max_correspondence_distance=params.icp.max_correspondence_distance,
            neighbors_pca=params.icp.neighbors_pca,
        )

        # Define initial pose for scan matcher = first pose recorded
        initial_pose = playback_data.step_data_list[0].true_pose

        # Init scan matcher
        scan_matcher = ScanMatcher(
            ogm=ogm,
            icp=icp,
            robo_param=(initial_pose, ...),  # wheel separation needed
            sensor_parameters=(
                map_data.sensor_param.min_sensor_range,
                map_data.sensor_param.max_sensor_range,
                params.scan_matcher.delta_r,
            ),
            occ_thres=params.scan_matcher.occ_thres,
        )
        
        return scan_matcher