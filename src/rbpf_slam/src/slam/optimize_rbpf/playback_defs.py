from typing import List, Tuple, Any
from dataclasses import dataclass, field

import numpy as np

from ..rbpf.scan_match_factory import OccupancyParams, SensorParams, MapParameter, ICPParams, RobotParams, ScanMatcherParams
from ..rbpf.rbpf import RBPFFactory, ParticleParams, MotionModelParams, MeasurementModelParams


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
class PlaybackData:
    '''
    Data storage for the whole playback run
    '''
    step_data_list: List[StepData]

@dataclass
class ExperimentParams:
    '''
    Parameters for the rbpf experiment
    '''
    occupancy_params: OccupancyParams
    sensor_params: SensorParams
    map_param: MapParameter
    icp_params: ICPParams
    robot_params: RobotParams
    scan_matcher_params: ScanMatcherParams
    particle_params: ParticleParams
    motion_model_params: MotionModelParams
    measurement_model_params: MeasurementModelParams
    tag: str 