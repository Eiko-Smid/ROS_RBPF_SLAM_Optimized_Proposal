#!/usr/bin/env python3
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from slam.infrastructure.defs import Pose2D

from slam.rbpf.particle import Particle
from slam.rbpf.motion_model import MotionModel
from slam.rbpf.measurement_model import MeasurementModel
from slam.rbpf.likelihood_filed_model import LikelihoodFiledModel

from slam.rbpf.proposal import ProposalEstimator
from slam.rbpf.resampler import Resampler
# from slam.scan_matcher.scan_matcher_factory import ScanMatcherFactory
from .scan_match_factory import (
    OccupancyParams,
    SensorParams,
    MapParameter,
    ICPParams,
    RobotParams,
    ScanMatcherParams,
    ScanMatchFactory,
)


@dataclass(frozen=True)
class ParticleParams:
    start_pose: Tuple[float, float, float]
    n_particles: int


@dataclass(frozen=True)
class MotionModelParams:
    sigma_x: float
    sigma_y: float
    sigma_theta: float
    wheel_separation: float
    ctrl_motion_fac: float
    ctrl_turn_fac: float


@dataclass(frozen=True)
class MeasurementModelParams:
    sigma_measurement: float
    every_nth_scan: int



class RBPFFactory():
    IDX_x=0
    IDX_y=1
    IDX_THETA=2

    def create(
            self,
            scan_match_fac: ScanMatchFactory,
            particle_params: ParticleParams,
            occ_param: OccupancyParams,
            sens_params: SensorParams,
            map_param: MapParameter,
            icp_params: ICPParams,
            robot_params: RobotParams,
            scan_matcher_params: ScanMatcherParams,
            motion_model_params: MotionModelParams,
            measurement_model_params: MeasurementModelParams,            
    ):
        # Init particle class
        particles = []
        w = 1/particle_params.n_particles

        for _ in range(particle_params.n_particles):
            scan_matcher = scan_match_fac.build(
                occ_param=occ_param,
                sens_params=sens_params,
                map_param=map_param,
                icp_params=icp_params,
                robo_param=robot_params,
                sm_params=scan_matcher_params,
            )

            pose: Pose2D = (
                particle_params.start_pose[self.IDX_x],
                particle_params.start_pose[self.IDX_y],
                particle_params.start_pose[self.IDX_THETA],
            )

            particles.append(
                Particle(
                    pose=pose,
                    weight=w,
                    scan_matcher=scan_matcher,
                )
            )   
        
        # Init motion model
        motion_model = MotionModel(
            sigma_x=motion_model_params.sigma_x,
            sigma_y=motion_model_params.sigma_y,
            sigma_theta=motion_model_params.sigma_theta,
            wheel_separation=motion_model_params.wheel_separation,
            ctrl_motion_fac=motion_model_params.ctrl_motion_fac,
            ctrl_turn_fac=motion_model_params.ctrl_turn_fac,
        )

        # init measurement model
        measurement_model = LikelihoodFiledModel(sigma=measurement_model_params.sigma_measurement)

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
            neff_threshold: Optional[float]= None,
    ):
        # Init RBPF memebers
        self.motion_model = motion_model
        self.measurement_model = measurement_model
        self.proposal = proposal
        self.resampler = resampler
        self.particles = particles
        
        # Define neff threshold for resampling
        if neff_threshold is not None:
            self.neff_threshold = neff_threshold
        else:
            self.neff_threshold = len(particles) / 2.0


    def weighted_mean_pose(self) -> Pose2D:
        '''
        Computes the weighted mean pose of the particle set. This can be used as an estimate for the current robot pose.

        Returns:
        --------
        Pose2D
            The weighted mean pose of the particle set.
        '''
        x = 0.0
        y = 0.0
        cos_theta = 0.0
        sin_theta = 0.0

        for p in self.particles:
            w = p.weight
            x += w * p.pose[0]
            y += w * p.pose[1]
            cos_theta += w * np.cos(p.pose[2])
            sin_theta += w * np.sin(p.pose[2])

        theta = np.arctan2(sin_theta, cos_theta)

        return (x, y, theta)
    

    @staticmethod
    def update_particle(
        particle: Particle,
        motion_model: MotionModel,
        measurement_model: MeasurementModel,
        proposal: ProposalEstimator,
        odom: Tuple[float, float],
        measurements: List[Tuple[float, float]],
        proposal_sigma_xy: float,
        proposal_sigma_theta: float,
        proposal_n_samples: int,
    ) -> Tuple[Particle, bool, bool]:
        '''
        Update step for a single particle. Updates the particle pose, weight and map based on the given odometry
        and measurements. Attention! Weights are not normalized!

        Including the following steps:
        1. Scan match particle pose with current measurements to get a corrected pose estimate.
        2. Compute optimized proposal distribution based on scan match pose and map points.
        3. If scan matching fails, fallback to motion model prediction and measurement model likelihood.
        4. Update map with new measurements and particle pose.

        Parameters:
        --------
        particle: Particle
            The particle to be updated.
        motion_model: MotionModel
            The motion model used for prediction and sampling.
        measurement_model: MeasurementModel
            The measurement model used for measurement likelihood estimation.
        proposal: ProposalEstimator
            The proposal estimator used for computing the optimized proposal distribution.
        odom: Tuple[float, float]
            The odometry measurements (dl, dr) for the current time step.
        measurements: List[Tuple[float, float]]
            The range measurements (range, bearing) for the current time step.

        Returns:
        --------        
        Particle
            The updated particle with new pose, weight (not normalized) and the updated map.
        '''
        # Information for debugging
        scan_match_failed = False
        scan_match_fallback_failed = False

        # Extract data
        dl, dr = odom

        # Scan match particle  
        corr_pose, pred_pose = particle.scan_matcher.update_pose(
            old_pose=particle.pose,
            dl=dl,
            dr=dr, 
            measurements=measurements,
        )

        # Get trained map points
        trained_nn_tree = particle.scan_matcher.get_trained_nn_tree()

        if corr_pose is not None:
            # Compute optimized proposal
            new_pose, p_weight = proposal.estimate_proposal(
                scan_match_pose=corr_pose,
                particle=particle,
                measurements=measurements,
                neighbor=trained_nn_tree,
                motion_model=motion_model,
                measurement_model=measurement_model,
                sigma_xy=proposal_sigma_xy,
                sigma_theta=proposal_sigma_theta,
                n_samples=proposal_n_samples,
            )
        # Fallback strategy if scan matching fails
        else:
            # Predict particle pose with motion model
            scan_match_failed = True
            dl_noisy, dr_noisy = motion_model.sample_noisy_ctrl(dl, dr)
            new_pose = motion_model.predict_pose(
                pose=pred_pose,
                dl=dl_noisy,
                dr=dr_noisy,
            )

            # Fallback to Measurement model with map points
            if trained_nn_tree is not None:
                # Compute particle weight
                p_weight = measurement_model.likelihood(
                    pose=pred_pose,
                    measurements=measurements,
                    scan_matcher= particle.scan_matcher,
                    neighbor=trained_nn_tree,                    
                )
            # Fallback strategy if scan matching fails
            else:
                scan_match_fallback_failed = True
                p_weight = 1.0
            
        # Update map
        # Extend map if necessary
        extension_needed = True
        while(extension_needed):
            extension_needed = particle.scan_matcher.ogm.map_extension_if_necessary(new_pose)
        # Update map
        particle.scan_matcher.ogm.update_map(
            measurements=measurements,
            pose=new_pose
        )

        new_particle = Particle(
            pose=new_pose,
            weight=particle.weight * p_weight,
            scan_matcher=particle.scan_matcher,
        )

        return new_particle, scan_match_failed, scan_match_fallback_failed


    def step(
        self,
        odom: Tuple[float, float],
        measurements: List[Tuple[float, float]],
        proposal_sigma_xy: float = 1.0,
        proposal_sigma_theta: float = 1.0,
        proposal_n_samples: int = 10,
    ) -> Tuple[float, bool, bool, Pose2D]:
        '''
        Performs the update step of the particle filter for all particles. This includes the following steps:
        1. Update each particle pose, weight and map based on the given odometry and measurements.
        2. Normalize particle weights.
        3. Resample particles if necessary based on the effective number of particles (neff).

        Parameters:
        --------
        odom: Tuple[float, float]
            The odometry measurements (dl, dr) for the current time step.
        measurements: List[Tuple[float, float]]
            The range measurements (range, bearing) for the current time step.
        
        Returns:
        --------
        float
            Effective number of particles (neff) computed from the normalized
            particle weights before optional resampling.
        '''
        # Process each particle
        scan_match_failed_any = False
        scan_match_fallback_failed_any = False

        for i, p in enumerate(self.particles):
            self.particles[i], scan_match_failed, scan_match_fallback_failed = self.update_particle(
                particle=p,
                motion_model=self.motion_model,
                measurement_model=self.measurement_model,
                proposal=self.proposal,
                odom=odom,
                measurements=measurements,
                proposal_sigma_xy=proposal_sigma_xy,
                proposal_sigma_theta=proposal_sigma_theta,
                proposal_n_samples=proposal_n_samples,
            )
            scan_match_failed_any = scan_match_failed_any or scan_match_failed
            scan_match_fallback_failed_any = scan_match_fallback_failed_any or scan_match_fallback_failed

        # Normalize particle weights
        # Normalize weights
        weights = np.array([p.weight for p in self.particles])
        norm = np.sum(weights)

        if norm == 0:
            # fallback: avoid division by zero
            norm_weights = np.ones(len(weights)) / len(weights)
        else:
            norm_weights = weights/norm

        # Update weights
        for i in range(len(self.particles)):
            self.particles[i].weight = norm_weights[i]

        # Compute live neff from current normalized weights before resampling.
        neff = float(self.resampler.compute_neff(norm_weights))

        # Best particle pose before optional resampling.
        best_idx = int(np.argmax(norm_weights))
        best_particle_pose = self.particles[best_idx].pose

        # Resampling
        # Check if resampling is necessary
        if neff < self.neff_threshold:
            # Get inidices of particles that have survived
            indices = self.resampler.low_variance_sampler(norm_weights)

            # Update particles
            new_partilces = []
            n_particles = len(self.particles)
            
            # Deep copy and update weight
            for idx in indices:
                p = self.particles[idx].copy()

                p.weight = 1.0 / n_particles

                new_partilces.append(p)

            # Replace old particle set by new set
            self.particles = new_partilces

        return neff, scan_match_failed_any, scan_match_fallback_failed_any, best_particle_pose
