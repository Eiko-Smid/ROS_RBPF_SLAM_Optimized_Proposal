#!/usr/bin/env python3
from typing import List, Tuple

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
        measurement_model = LikelihoodFiledModel(sigma_measurement)

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

    
    @staticmethod
    def update_particle(
        particle: Particle,
        motion_model: MotionModel,
        measurement_model: MeasurementModel,
        proposal: ProposalEstimator,
        odom: Tuple[float, float],
        measurements: List[Tuple[float, float]],
    ):
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
        # TODO: Fallback if scan matching failed
        else:
            # Predict particle pose with motion model
            dl_noisy, dr_noisy = motion_model.sample_noisy_ctrl(dl, dr)
            pred_pose = motion_model.predict_pose(
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
                    every_nth_measurement=5,
                )
            # Do MCL as fallback
            else:
                pass
            
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




    def step():
        pass
