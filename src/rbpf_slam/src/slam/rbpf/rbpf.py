#!/usr/bin/env python3
from typing import List, Tuple

from slam.infrastructure.defs import Pose2D

from slam.rbpf.particle import Particle
from slam.rbpf.motion_model import MotionModel
from slam.rbpf.measurement_model import MeasurementModel
from slam.rbpf.proposal import ProposalEstimator
from slam.rbpf.resampler import Resampler
from slam.scan_matcher.scan_matcher_factory import ScanMatcherFactory
from slam.scan_matcher.scan_match_playback_def import PlaybackData, ExperimentParams


class RBPF_Factory():
    IDX_x=0
    IDX_y=1
    IDX_THETA=2
    def __init__(self, scan_match_fac: ScanMatcherFactory):
        self.scan_match_fac = scan_match_fac


    def create(
            self,
            particle_params: Tuple[Pose2D, int],
            scan_match_params: Tuple[PlaybackData, ExperimentParams],
            motion_model_params: Tuple[float, float, float, float],
            measurement_model_params: Tuple[float],            
    ):
        # Extract params
        start_pose, n_particles = particle_params
        playback_data, exp_params = scan_match_params
        sigma_x, sigma_y, sigma_theta, wheel_separation = motion_model_params
        sigma_measurement = measurement_model_params[0]

        # Init particle class
        particles = []
        w = 1/n_particles

        for _ in range(n_particles):
            scan_matcher = self.scan_match_fac.build(
                playback_data=playback_data,
                params=exp_params
            )

            pose = Pose2D(
                x=start_pose[self.IDX_x],
                y=start_pose[self.IDX_y],
                theta=start_pose[self.IDX_THETA],
            )

            particles.append(
                Particle(
                    pose=pose,
                    weight=w,
                    scan_matcher=scan_matcher,
                )
            )   
        
        # Init motion model
        motion_model = MotionModel(sigma_x, sigma_y, sigma_theta, wheel_separation)

        # init measurement model
        measurement_model = MeasurementModel(sigma_measurement)

        # Init proposal Estimator
        proposal_estimator = ProposalEstimator()

        # init resampler
        resampler = Resampler()


        return RBPF(
            motion_model=motion_model,
            measurement_model=measurement_model,
            proposal=proposal_estimator,
            resampler=resampler,
            particles=particles,
        )



class RBPF:
    def __init__(
            self,
            motion_model: MotionModel,
            measurement_model: MeasurementModel,
            proposal: ProposalEstimator,
            resampler: Resampler,
            particles: List[Particle],

    ):
        self.motion_model = motion_model
        self.measurement_model = measurement_model
        self.proposal = proposal
        self.resampler = resampler
        self.particles = particles

    
    def update_particle():
        pass
        

    def step():
        pass
