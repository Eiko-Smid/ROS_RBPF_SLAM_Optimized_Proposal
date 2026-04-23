from typing import List, Tuple, Any
from dataclasses import dataclass, field

import numpy as np

@dataclass
class MeasurementModelParams:
    '''
    Parameters for the measurement model, including the measurement noise and the frequency of scan matching.
    '''
    sigma_measurement: float
    every_nth_scan: int


@dataclass
class MotionModelParams:
    '''
    Parameters for the motion model, including the noise parameters and control factors.
    '''
    sigma_x: float
    sigma_y: float
    sigma_theta: float
    wheel_separation: float
    ctrl_motion_fac: float
    ctrl_turn_fac: float


@dataclass
class ParticleParams:
    '''
    Parameters for the particle filter, including the initial pose and the number of particles.
    '''
    start_pose: Tuple[float, float, float]
    n_particles: int


@dataclass
class ScanMatchParams:
    '''
    Parameters for the scan matching, including the playback data and experiment parameters.
    '''
    playback_data: Any
    exp_params: Any 


@dataclass
class RBPFParams:
    '''
    Overall parameters for the RBPF, containing the motion model, measurement model, particle filter, and scan matching parameters.
    '''
    motion_model_params: MotionModelParams
    measurement_model_params: MeasurementModelParams
    particle_params: ParticleParams
    scan_match_params: ScanMatchParams