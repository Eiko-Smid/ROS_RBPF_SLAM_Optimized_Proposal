from typing import List, Tuple, Any
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StepData:
    '''
    Data storage to perform one step in scan matching 
    '''
    t: float
    dl: float
    dr: float
    scan: List[Tuple[float, float]]   # (range, bearing)
    true_pose: Tuple[float, float, float]  # (x, y, yaw)


@dataclass
class SensorParam:
    '''
    Sensor parameters for 2D LIDAR sensor.
    '''
    # min sensor range in meters
    min_sensor_range: float
    # max sensor range in meters
    max_sensor_range: float


@dataclass
class OccupancyParam:
    '''
    Metadata for occupancy grid mapping.
    '''
    prior_probability: float 
    increasing_probability: float
    decreasing_probability: float
    min_log_odds: float
    max_log_odds: float


@dataclass
class Metadata:
    '''
    Overall metadata for the playback dataset
    '''
    wheel_separation: float
    grid_resolution_m: float
    min_distance_to_border: float
    log_odds_map: np.ndarray
    sensor_param: SensorParam
    occupancy_param: OccupancyParam


@dataclass
class PlaybackData:
    '''
    Overall data structure for the playback dataset, containing metadata and step data list.
    '''
    meta_data: Metadata 
    step_data_list: List[StepData] = field(default_factory=list)


@dataclass(frozen=True)
class ICPParams:
    '''
    Parameters for ICP algorithm
    '''
    max_n_points: int
    max_correspondence_distance: float
    neighbors_pca: int
    max_iterations: int
    epsilon_rel: float
    no_improvement_limit: int
    min_error: float
    min_dtrans: float
    min_drot: float


@dataclass(frozen=True)
class ScanMatcherParams:
    '''
    Parameters for the scan matcher.
    '''
    occ_thres: float
    delta_r: float


@dataclass(frozen=True)
class ExperimentParams:
    '''
    Storage class for icp and scan match parameters 
    '''
    icp: ICPParams
    scan_matcher: ScanMatcherParams
    tag: str = ""