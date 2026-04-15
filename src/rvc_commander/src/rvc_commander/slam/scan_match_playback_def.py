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
class MapData:
    min_distance_to_border: float
    log_odds_map: np.ndarray
    sensor_param: SensorParam
    occupancy_param: OccupancyParam


@dataclass
class PlaybackData:
    map_data: MapData 
    step_data_list: List[StepData] = field(default_factory=list)
