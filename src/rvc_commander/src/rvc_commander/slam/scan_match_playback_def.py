from typing import List, Tuple, Any
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StepData:
    t: float
    dl: float
    dr: float
    scan: List[Tuple[float, float]]   # (range, bearing)
    true_pose: Tuple[float, float, float]  # (x, y, yaw)


@dataclass
class SensorParam:
    min_sensor_range: float
    max_sensor_range: float


@dataclass
class OccupancyParam:
    prior_probability: float 
    increasing_probability: float
    decreasing_probability: float
    min_log_odds: float
    max_log_odds: float


@dataclass
class Metadata:
    wheel_separation: float
    grid_resolution_m: float
    min_distance_to_border: float
    log_odds_map: np.ndarray
    sensor_param: SensorParam
    occupancy_param: OccupancyParam


@dataclass
class PlaybackData:
    map_data: Metadata 
    step_data_list: List[StepData] = field(default_factory=list)


@dataclass(frozen=True)
class ICPParams:
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
    occ_thres: float
    delta_r: float


@dataclass(frozen=True)
class ExperimentParams:
    icp: ICPParams
    scan_matcher: ScanMatcherParams
    tag: str = ""