#!/usr/bin/env python3
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time

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
            neff_threshold: Optional[float] = None,
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
            neff_threshold=neff_threshold,
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

        # Per-step metrics from the latest step call.
        self._step_counter = -1
        self._timing_stats = {
            "update_particles_sum_s": 0.0,
            "update_particles_count": 0,
            "normalize_neff_sum_s": 0.0,
            "normalize_neff_count": 0,
            "metrics_sum_s": 0.0,
            "metrics_count": 0,
            "resampling_sum_s": 0.0,
            "resampling_count": 0,
            "scan_match_update_pose_sum_s": 0.0,
            "scan_match_update_pose_count": 0,
            "proposal_estimation_sum_s": 0.0,
            "proposal_estimation_count": 0,
            "scan_match_fallback_sum_s": 0.0,
            "scan_match_fallback_count": 0,
            "map_extension_sum_s": 0.0,
            "map_extension_count": 0,
            "map_update_sum_s": 0.0,
            "map_update_count": 0,
        }
        self._timing_stats_scan_match_only = {
            "update_particle_sum_s": 0.0,
            "update_particle_count": 0,
            "scan_match_update_pose_sum_s": 0.0,
            "scan_match_update_pose_count": 0,
            "map_extension_sum_s": 0.0,
            "map_extension_count": 0,
            "map_update_sum_s": 0.0,
            "map_update_count": 0,
        }
        self._last_step_info = {
            "step": None,
            "neff": None,
            "true_pose": None,
            "scan_match_failed_any": None,
            "scan_match_fallback_failed_any": None,
            "best_particle_pose": None,
            "weighted_mean_pose": None,
            "particle_weight_min": None,
            "particle_weight_max": None,
            "particle_weight_mean": None,
            "timing_update_particles_s": None,
            "timing_normalize_neff_s": None,
            "timing_metrics_s": None,
            "timing_resampling_s": None,
            "proposal_metrics": None,
        }
        # Dedicated per-step info for scan-matching-only mode.
        self._last_step_info_scan_match_only = {
            "step": None,
            "neff": None,
            "true_pose": None,
            "scan_match_failed_any": None,
            "scan_match_fallback_failed_any": None,
            "best_particle_pose": None,
            "weighted_mean_pose": None,
            "particle_weight_min": None,
            "particle_weight_max": None,
            "particle_weight_mean": None,
            "timing_update_particles_s": None,
            "timing_normalize_neff_s": None,
            "timing_metrics_s": None,
            "timing_resampling_s": None,
            "proposal_metrics": None,
        }


    @staticmethod
    def _compute_weighted_mean_pose_from_particles(particles: List[Particle]) -> Pose2D:
        x = 0.0
        y = 0.0
        cos_theta = 0.0
        sin_theta = 0.0

        for p in particles:
            w = p.weight
            x += w * p.pose[0]
            y += w * p.pose[1]
            cos_theta += w * np.cos(p.pose[2])
            sin_theta += w * np.sin(p.pose[2])

        theta = np.arctan2(sin_theta, cos_theta)
        return (x, y, theta)


    def weighted_mean_pose(self) -> Pose2D:
        '''
        Computes the weighted mean pose of the particle set. This can be used as an estimate for the current robot pose.

        Returns:
        --------
        Pose2D
            The weighted mean pose of the particle set.
        '''
        # Use pre-resampling weighted mean from latest step if available.
        if self._last_step_info["weighted_mean_pose"] is not None:
            return self._last_step_info["weighted_mean_pose"]

        return self._compute_weighted_mean_pose_from_particles(self.particles)


    def get_step_info(self) -> dict:
        """
        Returns metrics from the latest step() call.
        """
        return self._last_step_info.copy()


    def get_step_info_scan_match_only(self) -> dict:
        """
        Returns metrics from the latest step_scan_match_only() call.
        """
        return self._last_step_info_scan_match_only.copy()


    def timing_summary(self) -> dict:
        """
        Returns end-of-run timing means for instrumented RBPF blocks.
        """
        return {
            "mean_timing_update_particles_s": self._safe_mean(
                self._timing_stats["update_particles_sum_s"], self._timing_stats["update_particles_count"]
            ),
            "mean_timing_normalize_neff_s": self._safe_mean(
                self._timing_stats["normalize_neff_sum_s"], self._timing_stats["normalize_neff_count"]
            ),
            "mean_timing_metrics_s": self._safe_mean(
                self._timing_stats["metrics_sum_s"], self._timing_stats["metrics_count"]
            ),
            "mean_timing_resampling_s": self._safe_mean(
                self._timing_stats["resampling_sum_s"], self._timing_stats["resampling_count"]
            ),
            "mean_timing_scan_match_update_pose_s": self._safe_mean(
                self._timing_stats["scan_match_update_pose_sum_s"], self._timing_stats["scan_match_update_pose_count"]
            ),
            "mean_timing_proposal_estimation_s": self._safe_mean(
                self._timing_stats["proposal_estimation_sum_s"], self._timing_stats["proposal_estimation_count"]
            ),
            "mean_timing_scan_match_fallback_s": self._safe_mean(
                self._timing_stats["scan_match_fallback_sum_s"], self._timing_stats["scan_match_fallback_count"]
            ),
            "mean_timing_map_extension_s": self._safe_mean(
                self._timing_stats["map_extension_sum_s"], self._timing_stats["map_extension_count"]
            ),
            "mean_timing_map_update_s": self._safe_mean(
                self._timing_stats["map_update_sum_s"], self._timing_stats["map_update_count"]
            ),
            "timing_update_particles_count": int(self._timing_stats["update_particles_count"]),
            "timing_normalize_neff_count": int(self._timing_stats["normalize_neff_count"]),
            "timing_metrics_count": int(self._timing_stats["metrics_count"]),
            "timing_resampling_count": int(self._timing_stats["resampling_count"]),
            "timing_scan_match_update_pose_count": int(self._timing_stats["scan_match_update_pose_count"]),
            "timing_proposal_estimation_count": int(self._timing_stats["proposal_estimation_count"]),
            "timing_scan_match_fallback_count": int(self._timing_stats["scan_match_fallback_count"]),
            "timing_map_extension_count": int(self._timing_stats["map_extension_count"]),
            "timing_map_update_count": int(self._timing_stats["map_update_count"]),
        }


    def timing_summary_scan_match_only(self) -> dict:
        """
        Returns end-of-run timing means for scan-matching-only blocks.
        """
        return {
            "mean_timing_sm_update_particle_s": self._safe_mean(
                self._timing_stats_scan_match_only["update_particle_sum_s"],
                self._timing_stats_scan_match_only["update_particle_count"],
            ),
            "mean_timing_sm_scan_match_update_pose_s": self._safe_mean(
                self._timing_stats_scan_match_only["scan_match_update_pose_sum_s"],
                self._timing_stats_scan_match_only["scan_match_update_pose_count"],
            ),
            "mean_timing_sm_map_extension_s": self._safe_mean(
                self._timing_stats_scan_match_only["map_extension_sum_s"],
                self._timing_stats_scan_match_only["map_extension_count"],
            ),
            "mean_timing_sm_map_update_s": self._safe_mean(
                self._timing_stats_scan_match_only["map_update_sum_s"],
                self._timing_stats_scan_match_only["map_update_count"],
            ),
            "timing_sm_update_particle_count": int(self._timing_stats_scan_match_only["update_particle_count"]),
            "timing_sm_scan_match_update_pose_count": int(self._timing_stats_scan_match_only["scan_match_update_pose_count"]),
            "timing_sm_map_extension_count": int(self._timing_stats_scan_match_only["map_extension_count"]),
            "timing_sm_map_update_count": int(self._timing_stats_scan_match_only["map_update_count"]),
        }


    @staticmethod
    def _safe_mean(sum_value: float, count: int) -> Optional[float]:
        if count <= 0:
            return None
        return float(sum_value) / float(count)
    

    def update_particle(
        self,
        particle: Particle,
        motion_model: MotionModel,
        measurement_model: MeasurementModel,
        proposal: ProposalEstimator,
        odom: Tuple[float, float],
        measurements_proposal: List[Tuple[float, float]],
        measurements_map_update: List[Tuple[float, float]],
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
        measurements_proposal: List[Tuple[float, float]]
            The range measurements (range, bearing) for the proposal step.
        measurements_map_update: List[Tuple[float, float]]
            The range measurements (range, bearing) for the map update step.

        Returns:
        --------        
        Particle
            The updated particle with new pose, weight (not normalized) and the updated map.
        '''
        # Set metrics to None
        prop_metrics = None 
        scan_match_failed = False
        scan_match_fallback_failed = False

        # Extract data
        dl, dr = odom

        # Scan match particle  
        t_scan_match_start = time.perf_counter()
        corr_pose, pred_pose = particle.scan_matcher.update_pose(
            old_pose=particle.pose,
            dl=dl,
            dr=dr, 
            measurements=measurements_proposal,
        )
        t_scan_match_s = time.perf_counter() - t_scan_match_start
        self._timing_stats["scan_match_update_pose_sum_s"] += t_scan_match_s
        self._timing_stats["scan_match_update_pose_count"] += 1

        # Get trained map points
        trained_nn_tree = particle.scan_matcher.get_trained_nn_tree()

        if corr_pose is not None:
            # Compute optimized proposal
            t_prop_start = time.perf_counter()
            new_pose, p_weight, prop_metrics = proposal.estimate_proposal(
                scan_match_pose=corr_pose,
                particle=particle,
                odom=odom,
                measurements=measurements_proposal,
                neighbor=trained_nn_tree,
                motion_model=motion_model,
                measurement_model=measurement_model,
                sigma_xy=proposal_sigma_xy,
                sigma_theta=proposal_sigma_theta,
                n_samples=proposal_n_samples,
            )
            t_prop_s = time.perf_counter() - t_prop_start
            self._timing_stats["proposal_estimation_sum_s"] += t_prop_s
            self._timing_stats["proposal_estimation_count"] += 1
        # Fallback strategy if scan matching fails
        else:
            t_fallback_start = time.perf_counter()

            # Predict particle pose with motion model
            # TODO: Check if noisy control needed or not. maybe simple run but with reduced noise
            scan_match_failed = True
            # dl_noisy, dr_noisy = motion_model.sample_noisy_ctrl(dl, dr)
            new_pose = motion_model.predict_pose(
                pose=particle.pose,
                dl=dl,
                dr=dr,
            )

            # Fallback to Measurement model with map points
            if trained_nn_tree is not None:
                # Compute particle weight
                p_weight = measurement_model.likelihood(
                    pose=new_pose,
                    measurements=measurements_proposal,
                    scan_matcher= particle.scan_matcher,
                    neighbor=trained_nn_tree,                    
                )
            # Fallback strategy if scan matching fails
            else:
                scan_match_fallback_failed = True
                # TODO: Maybe set the lower to punish particle if no prob could be computed 
                p_weight = 1.0

            t_fallback_s = time.perf_counter() - t_fallback_start
            self._timing_stats["scan_match_fallback_sum_s"] += t_fallback_s
            self._timing_stats["scan_match_fallback_count"] += 1
            
        # Update map
        # Extend map if necessary
        t_map_ext_start = time.perf_counter()
        extension_needed = True
        while(extension_needed):
            extension_needed = particle.scan_matcher.ogm.map_extension_if_necessary(new_pose)
        t_map_ext_s = time.perf_counter() - t_map_ext_start
        self._timing_stats["map_extension_sum_s"] += t_map_ext_s
        self._timing_stats["map_extension_count"] += 1
            
        # Update map
        t_map_update_start = time.perf_counter()
        particle.scan_matcher.ogm.update_map(
            measurements=measurements_map_update,
            pose=new_pose
        )
        t_map_update_s = time.perf_counter() - t_map_update_start
        self._timing_stats["map_update_sum_s"] += t_map_update_s
        self._timing_stats["map_update_count"] += 1

        new_particle = Particle(
            pose=new_pose,
            weight=particle.weight * p_weight,
            scan_matcher=particle.scan_matcher,
        )

        return new_particle, scan_match_failed, scan_match_fallback_failed, prop_metrics


    def step(
        self,
        odom: Tuple[float, float],
        measurements_proposal: List[Tuple[float, float]],
        measurements_map_update: List[Tuple[float, float]],
        true_pose: Optional[Pose2D] = None,
        proposal_sigma_xy: float = 1.0,
        proposal_sigma_theta: float = 1.0,
        proposal_n_samples: int = 10,
    ) -> Tuple[float, Pose2D]:
        '''
        Performs the update step of the particle filter for all particles. This includes the following steps:
        1. Update each particle pose, weight and map based on the given odometry and measurements.
        2. Normalize particle weights.
        3. Resample particles if necessary based on the effective number of particles (neff).

        We got to different values for the measurements. measurements_proposal is used for the scan matching and 
        proposal distribution estimation, while measurements_map_update is used for updating the map. 
        This allows to use different measurement subsets for the different steps, e.g. using a subsampled scan for 
        the scan matching and proposal estimation, while using the full scan for the map update.

        Parameters:
        --------
        odom: Tuple[float, float]
            The odometry measurements (dl, dr) for the current time step.
        measurements_proposal: List[Tuple[float, float]]
            The range measurements (range, bearing) for the proposal step.
        measurements_map_update: List[Tuple[float, float]]
            The range measurements (range, bearing) for the map update step.
        true_pose: Optional[Pose2D]
            The true pose of the robot for the current time step. This is only used for logging
            and debugging purposes, e.g. to compute the error of the best particle pose and the weighted mean pose.
        proposal_sigma_xy: float
            The standard deviation in x and y direction for the optimized proposal distribution.
        proposal_sigma_theta: float
            The standard deviation for the orientation for the optimized proposal distribution.
        proposal_n_samples: int
            The number of samples to draw from the optimized proposal distribution for each particle.
        
        Returns:
        --------
        Tuple[float, Pose2D]
            The effective number of particles (neff) and the weighted mean pose before resampling.
        '''
        
        scan_match_failed_any = False
        scan_match_fallback_failed_any = False
        particle0_prop_metrics = None
        self._step_counter += 1
        step_idx = self._step_counter

        # Process each particle
        # t_update_start = time.perf_counter()
        for i, p in enumerate(self.particles):
            self.particles[i], scan_match_failed, scan_match_fallback_failed, prop_metrics = self.update_particle(
                particle=p,
                motion_model=self.motion_model,
                measurement_model=self.measurement_model,
                proposal=self.proposal,
                odom=odom,
                measurements_proposal=measurements_proposal,
                measurements_map_update=measurements_map_update,
                proposal_sigma_xy=proposal_sigma_xy,
                proposal_sigma_theta=proposal_sigma_theta,
                proposal_n_samples=proposal_n_samples,
            )
            scan_match_failed_any = scan_match_failed_any or scan_match_failed
            scan_match_fallback_failed_any = scan_match_fallback_failed_any or scan_match_fallback_failed
            if i == 0:
                particle0_prop_metrics = prop_metrics
        # t_update_s = time.perf_counter() - t_update_start
        # self._timing_stats["update_particles_sum_s"] += t_update_s
        # self._timing_stats["update_particles_count"] += 1
        t_update_s = None

        # Normalize particle weights
        # Normalize weights
        # t_norm_start = time.perf_counter()
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
        # t_norm_s = time.perf_counter() - t_norm_start
        # self._timing_stats["normalize_neff_sum_s"] += t_norm_s
        # self._timing_stats["normalize_neff_count"] += 1
        t_norm_s = None

        # Compute metrics
        # t_metrics_start = time.perf_counter()
        # Weighted mean pose before optional resampling.
        weighted_mean_pose = self._compute_weighted_mean_pose_from_particles(self.particles)
        # Get best particle pose before resampling
        best_idx = int(np.argmax(norm_weights))
        best_particle_pose = self.particles[best_idx].pose
        # Weight statistics before optional resampling.
        particle_weight_min = float(np.min(norm_weights))
        particle_weight_max = float(np.max(norm_weights))
        particle_weight_mean = float(np.mean(norm_weights))
        # t_metrics_s = time.perf_counter() - t_metrics_start
        # self._timing_stats["metrics_sum_s"] += t_metrics_s
        # self._timing_stats["metrics_count"] += 1
        t_metrics_s = None

        t_resampling_s = None

        # Store step metrics before any potential resampling mutates particle set.
        self._last_step_info = {
            "step": step_idx,
            "neff": neff,
            "true_pose": true_pose,
            "scan_match_failed_any": scan_match_failed_any,
            "scan_match_fallback_failed_any": scan_match_fallback_failed_any,
            "best_particle_idx": best_idx,
            "best_particle_pose": best_particle_pose,
            "best_particle_map": self.particles[best_idx].scan_matcher.ogm.return_log_odds_map(),
            "weighted_mean_pose": weighted_mean_pose,
            "particle_weight_min": particle_weight_min,
            "particle_weight_max": particle_weight_max,
            "particle_weight_mean": particle_weight_mean,
            "timing_update_particles_s": t_update_s,
            "timing_normalize_neff_s": t_norm_s,
            "timing_metrics_s": t_metrics_s,
            "timing_resampling_s": t_resampling_s,
            "proposal_metrics": particle0_prop_metrics,
        }

        # Resampling
        # Check if resampling is necessary
        if neff < self.neff_threshold:
            # t_resampling_start = time.perf_counter()
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

            # t_resampling_s = time.perf_counter() - t_resampling_start
            # self._timing_stats["resampling_sum_s"] += t_resampling_s
            # self._timing_stats["resampling_count"] += 1

        # Fill resampling timings after optional resampling.
        self._last_step_info["timing_resampling_s"] = t_resampling_s

        return neff, weighted_mean_pose


    def update_particle_scan_match_only(
        self,
        particle: Particle,
        odom: Tuple[float, float],
        measurements_filter: List[Tuple[float, float]],
        measurements_map_update: List[Tuple[float, float]],
    ) -> Tuple[Particle, bool]:
        """
        Scan-matching-only update for a single particle.

        This path does not perform proposal estimation or measurement-model weighting.
        It predicts/corrects pose with the scan matcher and updates the map only when
        a corrected pose is available.
        """
        scan_match_failed = False

        dl, dr = odom

        t_scan_match_start = time.perf_counter()
        corr_pose, pred_pose = particle.scan_matcher.update_pose(
            old_pose=particle.pose,
            dl=dl,
            dr=dr,
            measurements=measurements_filter,
        )
        t_scan_match_s = time.perf_counter() - t_scan_match_start
        self._timing_stats_scan_match_only["scan_match_update_pose_sum_s"] += t_scan_match_s
        self._timing_stats_scan_match_only["scan_match_update_pose_count"] += 1

        # Keep trajectory continuity using prediction, but only map with corrected pose.
        new_pose = corr_pose if corr_pose is not None else pred_pose

        if corr_pose is None:
            scan_match_failed = True

        # Keep map growth in scan-only mode even when scan matching fails.
        t_map_ext_start = time.perf_counter()
        extension_needed = True

        while extension_needed:
            extension_needed = particle.scan_matcher.ogm.map_extension_if_necessary(new_pose)

        t_map_ext_s = time.perf_counter() - t_map_ext_start
        self._timing_stats_scan_match_only["map_extension_sum_s"] += t_map_ext_s
        self._timing_stats_scan_match_only["map_extension_count"] += 1

        t_map_update_start = time.perf_counter()
        particle.scan_matcher.ogm.update_map(
            measurements=measurements_map_update,
            pose=new_pose,
        )
        t_map_update_s = time.perf_counter() - t_map_update_start
        self._timing_stats_scan_match_only["map_update_sum_s"] += t_map_update_s
        self._timing_stats_scan_match_only["map_update_count"] += 1
        self._last_timing_sm_map_update_s = t_map_update_s

        new_particle = Particle(
            pose=new_pose,
            weight=particle.weight,
            scan_matcher=particle.scan_matcher,
        )

        return new_particle, scan_match_failed


    def step_scan_match_only(
        self,
        odom: Tuple[float, float],
        measurements_filter: List[Tuple[float, float]],
        measurements_map_update: List[Tuple[float, float]],
    ) -> Tuple[float, Pose2D]:
        """
        Runs one RBPF step in scan-matching-only mode.

        This mode skips proposal and measurement model updates and is intended for
        focused scan matcher diagnostics.
        """
       
        self._step_counter += 1
        step_idx = self._step_counter

        # Update particle based on scan matcher
        t_update_particle_start = time.perf_counter()
        self.particles[0], scan_match_failed = self.update_particle_scan_match_only(
            particle=self.particles[0],
            odom=odom,
            measurements_filter=measurements_filter,
            measurements_map_update=measurements_map_update,
        )
        t_update_particle_s = time.perf_counter() - t_update_particle_start
        self._timing_stats_scan_match_only["update_particle_sum_s"] += t_update_particle_s
        self._timing_stats_scan_match_only["update_particle_count"] += 1
    
        self._last_step_info_scan_match_only = {
            "step": step_idx,
            "scan_match_failed": scan_match_failed,            
            "particle_pose": self.particles[0].pose,
            "particle_map": self.particles[0].scan_matcher.ogm.return_log_odds_map(),
            "timing_update_particle": t_update_particle_s,
            "timing_ogm_update": getattr(self, "_last_timing_sm_map_update_s", None),
        }

        # Keep compatibility for callers still using get_step_info().
        self._last_step_info = dict(self._last_step_info_scan_match_only)

    
    def update_particle_without_proposal_pose(
        self,
        particle: Particle,
        motion_model: MotionModel,
        measurement_model: MeasurementModel,
        proposal: ProposalEstimator,
        odom: Tuple[float, float],
        measurements_proposal: List[Tuple[float, float]],
        measurements_map_update: List[Tuple[float, float]],
        proposal_sigma_xy: float,
        proposal_sigma_theta: float,
        proposal_n_samples: int,
    ) -> Tuple[Particle, bool, bool]:
        '''
        Update step for a single particle. Updates the particle pose, weight and map based on the given odometry
        and measurements. Attention! Weights are not normalized!
        This variant doesn't use the sampled pose from the Proposal, it uses the scan matcher pose instead. However 
        the  

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
        measurements_proposal: List[Tuple[float, float]]
            The range measurements (range, bearing) for the proposal step.
        measurements_map_update: List[Tuple[float, float]]
            The range measurements (range, bearing) for the map update step.

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
        t_scan_match_start = time.perf_counter()
        corr_pose, pred_pose = particle.scan_matcher.update_pose(
            old_pose=particle.pose,
            dl=dl,
            dr=dr, 
            measurements=measurements_proposal,
        )

        t_scan_match_s = time.perf_counter() - t_scan_match_start
        self._timing_stats["scan_match_update_pose_sum_s"] += t_scan_match_s
        self._timing_stats["scan_match_update_pose_count"] += 1

        # Get trained map points
        trained_nn_tree = particle.scan_matcher.get_trained_nn_tree()

        if corr_pose is not None:
            # Compute optimized proposal
            t_prop_start = time.perf_counter()
            _, p_weight = proposal.estimate_proposal(
                scan_match_pose=corr_pose,
                particle=particle,
                odom=odom,
                measurements=measurements_proposal,
                neighbor=trained_nn_tree,
                motion_model=motion_model,
                measurement_model=measurement_model,
                sigma_xy=proposal_sigma_xy,
                sigma_theta=proposal_sigma_theta,
                n_samples=proposal_n_samples,
            )

            new_pose = corr_pose

            t_prop_s = time.perf_counter() - t_prop_start
            self._timing_stats["proposal_estimation_sum_s"] += t_prop_s
            self._timing_stats["proposal_estimation_count"] += 1
        # Fallback strategy if scan matching fails
        else:
            t_fallback_start = time.perf_counter()

            # Predict particle pose with motion model
            scan_match_failed = True
            # dl_noisy, dr_noisy = motion_model.sample_noisy_ctrl(dl, dr)
            new_pose = motion_model.predict_pose(
                pose=particle.pose,
                dl=dl,
                dr=dr,
            )

            # Fallback to Measurement model with map points
            if trained_nn_tree is not None:
                # Compute particle weight
                p_weight = measurement_model.likelihood(
                    pose=new_pose,
                    measurements=measurements_proposal,
                    scan_matcher= particle.scan_matcher,
                    neighbor=trained_nn_tree,                    
                )
            # Fallback strategy if scan matching fails
            else:
                scan_match_fallback_failed = True
                # TODO: Maybe set the lower to punish particle if no prob could be computed 
                p_weight = 1.0

            t_fallback_s = time.perf_counter() - t_fallback_start
            self._timing_stats["scan_match_fallback_sum_s"] += t_fallback_s
            self._timing_stats["scan_match_fallback_count"] += 1
            
        # Update map
        # Extend map if necessary
        t_map_ext_start = time.perf_counter()
        extension_needed = True

        while(extension_needed):
            extension_needed = particle.scan_matcher.ogm.map_extension_if_necessary(new_pose)

        t_map_ext_s = time.perf_counter() - t_map_ext_start
        self._timing_stats["map_extension_sum_s"] += t_map_ext_s
        self._timing_stats["map_extension_count"] += 1
            
        # Update map
        t_map_update_start = time.perf_counter()
        particle.scan_matcher.ogm.update_map(
            measurements=measurements_map_update,
            pose=new_pose
        )
        t_map_update_s = time.perf_counter() - t_map_update_start
        self._timing_stats["map_update_sum_s"] += t_map_update_s
        self._timing_stats["map_update_count"] += 1

        new_particle = Particle(
            pose=new_pose,
            weight=particle.weight * p_weight,
            scan_matcher=particle.scan_matcher,
        )

        return new_particle, scan_match_failed, scan_match_fallback_failed

    
    def step_rbpf_without_proposal_pose(
        self,
        odom: Tuple[float, float],
        measurements_proposal: List[Tuple[float, float]],
        measurements_map_update: List[Tuple[float, float]],
        true_pose: Optional[Pose2D] = None,
        proposal_sigma_xy: float = 1.0,
        proposal_sigma_theta: float = 1.0,
        proposal_n_samples: int = 10,
    ):
        scan_match_failed_any = False
        scan_match_fallback_failed_any = False
        self._step_counter += 1
        step_idx = self._step_counter

        # Process each particle
        # t_update_start = time.perf_counter()
        for i, p in enumerate(self.particles):
            self.particles[i], scan_match_failed, scan_match_fallback_failed = self.update_particle_without_proposal_pose(
                particle=p,
                motion_model=self.motion_model,
                measurement_model=self.measurement_model,
                proposal=self.proposal,
                odom=odom,
                measurements_proposal=measurements_proposal,
                measurements_map_update=measurements_map_update,
                proposal_sigma_xy=proposal_sigma_xy,
                proposal_sigma_theta=proposal_sigma_theta,
                proposal_n_samples=proposal_n_samples,
            )
            scan_match_failed_any = scan_match_failed_any or scan_match_failed
            scan_match_fallback_failed_any = scan_match_fallback_failed_any or scan_match_fallback_failed

        t_update_s = None

        # Normalize particle weights
        # Normalize weights
        # t_norm_start = time.perf_counter()
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
        # t_norm_s = time.perf_counter() - t_norm_start
        # self._timing_stats["normalize_neff_sum_s"] += t_norm_s
        # self._timing_stats["normalize_neff_count"] += 1
        t_norm_s = None

        # Compute metrics
        # t_metrics_start = time.perf_counter()
        # Weighted mean pose before optional resampling.
        weighted_mean_pose = self._compute_weighted_mean_pose_from_particles(self.particles)
        # Get best particle pose before resampling
        best_idx = int(np.argmax(norm_weights))
        best_particle_pose = self.particles[best_idx].pose
        # Weight statistics before optional resampling.
        particle_weight_min = float(np.min(norm_weights))
        particle_weight_max = float(np.max(norm_weights))
        particle_weight_mean = float(np.mean(norm_weights))
        # t_metrics_s = time.perf_counter() - t_metrics_start
        # self._timing_stats["metrics_sum_s"] += t_metrics_s
        # self._timing_stats["metrics_count"] += 1
        t_metrics_s = None

        t_resampling_s = None

        # Store step metrics before any potential resampling mutates particle set.
        self._last_step_info = {
            "step": step_idx,
            "neff": neff,
            "true_pose": true_pose,
            "scan_match_failed_any": scan_match_failed_any,
            "scan_match_fallback_failed_any": scan_match_fallback_failed_any,
            "best_particle_idx": best_idx,
            "best_particle_pose": best_particle_pose,
            "best_particle_map": self.particles[best_idx].scan_matcher.ogm.return_log_odds_map(),
            "weighted_mean_pose": weighted_mean_pose,
            "particle_weight_min": particle_weight_min,
            "particle_weight_max": particle_weight_max,
            "particle_weight_mean": particle_weight_mean,
            "timing_update_particles_s": t_update_s,
            "timing_normalize_neff_s": t_norm_s,
            "timing_metrics_s": t_metrics_s,
            "timing_resampling_s": t_resampling_s,
        }

        # Resampling
        # Check if resampling is necessary
        if neff < self.neff_threshold:
            # t_resampling_start = time.perf_counter()
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

            # t_resampling_s = time.perf_counter() - t_resampling_start
            # self._timing_stats["resampling_sum_s"] += t_resampling_s
            # self._timing_stats["resampling_count"] += 1

        # Fill resampling timings after optional resampling.
        self._last_step_info["timing_resampling_s"] = t_resampling_s

        return neff, weighted_mean_pose
