#!/usr/bin/env python3
from typing import List, Tuple, Optional

import numpy as np

from slam.infrastructure.defs import Pose2D

from slam.rbpf.particle import Particle
from slam.rbpf.motion_model import MotionModel
from slam.rbpf.measurement_model import MeasurementModel
from slam.rbpf.likelihood_filed_model import LikelihoodFiledModel

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
            motion_model_params: Tuple[float, float, float, float, float, float],
            measurement_model_params: Tuple[float, int],            
    ):
        # Extract params
        start_pose, n_particles = particle_params
        playback_data, exp_params = scan_match_params
        sigma_x, sigma_y, sigma_theta, wheel_separation, ctrl_motion_fac, ctrl_turn_fac = motion_model_params
        sigma_measurement, every_nth_scan = measurement_model_params[0]

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
        motion_model = MotionModel(
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            sigma_theta=sigma_theta,
            wheel_separation=wheel_separation,
            ctrl_motion_fac=ctrl_motion_fac,
            ctrl_turn_fac=ctrl_turn_fac,
        )

        # init measurement model
        measurement_model = LikelihoodFiledModel(sigma=sigma_measurement)

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


    @staticmethod
    def update_particle(
        particle: Particle,
        motion_model: MotionModel,
        measurement_model: MeasurementModel,
        proposal: ProposalEstimator,
        odom: Tuple[float, float],
        measurements: List[Tuple[float, float]],
    ) -> Particle:
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
            )
        # Fallback strategy if scan matching fails
        else:
            # Predict particle pose with motion model
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

        return Particle(
            pose=new_pose,
            weight=particle.weight * p_weight,
            scan_matcher=particle.scan_matcher,
        )


    def step(self, odom: Tuple[float, float], measurements: List[Tuple[float, float]]) -> None:
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
        None.
        '''
        # Process each particle
        for i, p in enumerate(self.particles):
            self.particles[i] = self.update_particle(
                particle=p,
                motion_model=self.motion_model,
                measurement_model=self.measurement_model,
                proposal=self.proposal,
                odom=odom,
                measurements=measurements
            )

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

        # Resampling
        # Check if resampling is necessary
        if self.resampler.do_resampling(
            weights=norm_weights,
            min_neff=self.neff_threshold,
        ):
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
