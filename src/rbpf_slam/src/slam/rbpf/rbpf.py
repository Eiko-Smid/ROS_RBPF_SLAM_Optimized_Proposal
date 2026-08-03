#!/usr/bin/env python3
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
from enum import Enum
import time

import numpy as np

from slam.infrastructure.defs import Pose2D

from slam.rbpf.particle import Particle
from slam.rbpf.motion_model import MotionModel
from slam.rbpf.measurement_model import MeasurementModel
from slam.rbpf.likelihood_filed_model import LikelihoodFiledModel
from slam.rbpf.beam_range_finder_model import BeamRangeFinderModel

from slam.rbpf.proposal import ProposalEstimator
from slam.rbpf.resampler import Resampler

from .particle_process_pool import ParticleProcessPool
# from . import particle_process_pool as worker_state


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

'''
Future TODos

Delete the following methods:

    - update_particle_without_proposal_pose
    - step_rbpf_without_proposal_pose
    - step
    - update_particle

    - Rename the current sued methods if possible


'''


INVALID_LOG_LIKELIHOOD = -1.0e12
FALLBACK_LOG_FLOOR = -80.0
RNG_STREAM_PROPOSAL = 0
RNG_STREAM_RESAMPLING = 1


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


@dataclass(frozen=True)
class BeamRangeFinderMeasModelParams:
    occ_thresh: float = 1.4
    free_thresh: float = -1.4
    unknown_ratio_thresh: float = 0.30
    known_free_ratio_thresh: float = 0.70

    sigma_hit: float = 0.15
    w_hit: float = 0.70   
    w_short: float = 0.10
    lambda_short: float = 0.20
    w_max: float = 0.10
    w_rand: float = 0.10
    
    p_unknown: float = 0.20
    p_out_of_map: float = 0.10
    p_unexpected_known_free: float = 0.03
    p_pred_below_min: float = 0.02

    # Numerical / scaling
    alpha_meas: float = 0.10
    beam_step: int = 2  
    eps: float = 1e-12


# Dataclass including the task load for the process pool in order to process partciles in parallel
@dataclass(frozen=True)
class ParticleUpdateTask:
    particle_index: int
    particle: Particle
    motion_model: MotionModel
    measurement_model: MeasurementModel

    odom: Tuple[float, float]
    measurements_proposal: List[Tuple[float, float]]
    measurements_map_update: List[Tuple[float, float]]
    proposal_seed: Optional[int] = None
    proposal_sigma_xy: float = 1.0
    proposal_sigma_theta: float = 1.0
    proposal_n_samples: int = 10
    cov_std_scale: float = 0.5
    cov_max_std_xy: float = 0.02
    cov_max_std_theta: float = 0.02
    min_std_xy: float = 0.0
    min_std_theta: float = 0.0


@dataclass(frozen=True)
class ParticleUpdateResult:
    particle_index: int
    updated_particle: Particle
    log_p_weight: float
    scan_match_failed: bool
    scan_match_fallback_failed: bool
    meas_model_fallback_res: Optional[dict] = None
    prop_metrics: Optional[dict] = None


def _derive_operation_seed(
    run_seed: Optional[int],
    step_idx: int,
    stream_id: int,
    particle_index: int = 0,
) -> Optional[int]:
    """Derive a stable seed for one logical random operation."""
    if run_seed is None:
        return None

    if run_seed < 0:
        raise ValueError(f"run_seed must be non-negative, got {run_seed}.")

    seed_sequence = np.random.SeedSequence(
        [
            int(run_seed),
            int(step_idx),
            int(stream_id),
            int(particle_index),
        ]
    )
    return int(seed_sequence.generate_state(1)[0])


def update_particle_worker(
    task: ParticleUpdateTask
) -> ParticleUpdateResult:
    proposal_rng = (
        np.random.default_rng(task.proposal_seed)
        if task.proposal_seed is not None
        else None
    )

    # Extract task
    particle: Particle = task.particle
    motion_model: MotionModel = task.motion_model
    meas_model: MeasurementModel = task.measurement_model
    odom: Tuple[float, float] = task.odom
    measurements_proposal: List[Tuple[float, float]] = task.measurements_proposal
    measurements_map_update: List[Tuple[float, float]] = task.measurements_map_update
    proposal_sigma_xy: float = task.proposal_sigma_xy
    proposal_sigma_theta: float = task.proposal_sigma_theta
    proposal_n_samples: int = task.proposal_n_samples
    cov_std_scale: float = task.cov_std_scale
    cov_max_std_xy: float = task.cov_max_std_xy
    cov_max_std_theta: float = task.cov_max_std_theta
    min_std_xy: float = task.min_std_xy
    min_std_theta: float = task.min_std_theta

    # Init proposal
    proposal = ProposalEstimator()

    # Set metrics to None
    prop_metrics = None 
    scan_match_failed = False
    scan_match_fallback_failed = False

    # Init measurement model fallback counters for diagnostics
    meas_model_fallback_res = {
        "log_likelihood": 0.0,
        "mean_abs_error": 0.0,
        "valid_beam_count": 0,
        "map_hit_count": 0,
        "no_map_hit_count": 0,
        "out_of_map_count": 0,
        "unknown_ray_count": 0,
        "known_free_ray_count": 0,
        "unexpected_known_free_count": 0,
        "skipped_beam_count": 0,
    }

    # Extract data
    dl, dr = odom

    # Scan match particle  
    corr_pose, pred_pose = particle.scan_matcher.update_pose(
        old_pose=particle.pose,
        dl=dl,
        dr=dr, 
        measurements=measurements_proposal,
    )

    # Compute proposal if scan matching was successful
    if corr_pose is not None:
        # Compute optimized proposal    
        new_pose, log_p_weight, prop_metrics = proposal.estimate_proposal_range_finder(
            scan_match_pose=corr_pose,
            particle=particle,
            odom=odom,
            measurements=measurements_proposal,
            motion_model=motion_model,
            measurement_model=meas_model,
            sigma_xy=proposal_sigma_xy,
            sigma_theta=proposal_sigma_theta,
            n_samples=proposal_n_samples,
            cov_std_scale=cov_std_scale,
            cov_max_std_xy=cov_max_std_xy,
            cov_max_std_theta=cov_max_std_theta,
            min_std_xy=min_std_xy,
            min_std_theta=min_std_theta,
            rng=proposal_rng,
        )

    # Fallback strategy if scan matching failed
    else:
        # Predict particle pose with motion model
        scan_match_failed = True

        new_pose = motion_model.predict_pose(
            pose=particle.pose,
            dl=dl,
            dr=dr,
        )

        # Fallback to Measurement model 
        meas_model_fallback_res = meas_model.likelihood(
            pose=new_pose,
            measurements=measurements_proposal,
            ogm=particle.scan_matcher.ogm,
        )

        # Extract likilihood 
        log_p_weight = meas_model_fallback_res.get("log_likelihood", FALLBACK_LOG_FLOOR)
        
        if not np.isfinite(log_p_weight):
            scan_match_fallback_failed = True
            log_p_weight = FALLBACK_LOG_FLOOR
    
    # Update map
    # Extend map if necessary
    extension_needed = True
    while(extension_needed):
        extension_needed = particle.scan_matcher.ogm.map_extension_if_necessary(new_pose)

    # Update map
    particle.scan_matcher.ogm.update_map(
        measurements=measurements_map_update,
        pose=new_pose
    )
    
    # Update particle
    new_particle = Particle(
        pose=new_pose,
        weight=particle.weight,
        scan_matcher=particle.scan_matcher,
    )

    # Update proposal metrics
    if prop_metrics is None:
        prop_metrics = {}
    prop_metrics["scan_matcher_info"] = particle.scan_matcher.get_info()

    
    result = ParticleUpdateResult(
        particle_index=task.particle_index,
        updated_particle=new_particle,
        log_p_weight=log_p_weight,
        meas_model_fallback_res=meas_model_fallback_res,

        scan_match_failed=scan_match_failed,
        scan_match_fallback_failed=scan_match_fallback_failed,
        prop_metrics=prop_metrics
    )

    return result


    
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
            measurement_model_params: Union[MeasurementModelParams, BeamRangeFinderMeasModelParams],
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

        # Backward-compatible dispatch for both measurement-model parameter types.
        if isinstance(measurement_model_params, BeamRangeFinderMeasModelParams):
            measurement_model = BeamRangeFinderModel(**vars(measurement_model_params))
        else:
            measurement_model = LikelihoodFiledModel(
                sigma=measurement_model_params.sigma_measurement
            )

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


class InitStatus(Enum):
    WAITING = "waiting for initialization"
    INITIALIZING = "Initialization in progress"
    SUCCESS = "Init successful"
    FAILED_ODOM_THRESHOLD = (
        "Init failed/skipped because dl or dr exceeded the odometry threshold. "
        "Initialization will never be executed again."
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

        # Initialization
        # Number of initial data steps that were already consumed by init_process().
        # Starts at 0 because no initialization scan has been processed yet.
        self.init_counter = 0
        self.init_status = InitStatus.WAITING
        self.init_count_threshold = 4

        # If abs(dl) or abs(dr) exceeds this value before initialization finished,
        # initialization is skipped forever and scan-match-only mode starts directly.
        self.odom_threshold = 0.005
        self.init_failure_reason = None
        
        # Define neff threshold for resampling
        if neff_threshold is not None:
            self.neff_threshold = neff_threshold
        else:
            self.neff_threshold = len(particles) / 2.0

        # Per-step metrics from the latest step call.
        self._step_counter = -1

        # Increments whenever the filter updates the particles. Since the particles are not updated in the 
        # initialization process, this counter + self.init_counter = total filter steps.
        self.particle_update_counter = 0

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
            "t_init_process_sum_s": 0.0,
            "t_init_process_count": 0,
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


    def time_summary_scan_match_only(self) -> dict:
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

    
    def _stop_init_process(self, odom: Tuple[float, float]) -> bool:
        """
        Returns True if initialization must be skipped because the robot already moved
        too much according to wheel odometry.
        """
        dl, dr = odom

        stop_cond = (abs(dl) > self.odom_threshold or abs(dr) > self.odom_threshold)
        return stop_cond

    
    def init_process(
        self,
        particle: Particle,
        measurements_map_update: List[Tuple[float, float]],
    ) -> Particle:        
        # Extend map if necessary
        t_map_ext_start = time.perf_counter()
        extension_needed = True
        while extension_needed:
            extension_needed = particle.scan_matcher.ogm.map_extension_if_necessary(particle.pose)
        t_map_ext_s = time.perf_counter() - t_map_ext_start
        self._timing_stats_scan_match_only["map_extension_sum_s"] += t_map_ext_s
        self._timing_stats_scan_match_only["map_extension_count"] += 1

        # Update map at the current unchanged particle pose.
        # Important: initialization does NOT run scan matching and does NOT move the pose.
        t_map_update_start = time.perf_counter()
        particle.scan_matcher.ogm.update_map(
            measurements=measurements_map_update,
            pose=particle.pose,
        )
        t_map_update_s = time.perf_counter() - t_map_update_start
        self._timing_stats_scan_match_only["map_update_sum_s"] += t_map_update_s
        self._timing_stats_scan_match_only["map_update_count"] += 1
        self._last_timing_sm_map_update_s = t_map_update_s

        new_particle = Particle(
            pose=particle.pose,
            weight=particle.weight,
            scan_matcher=particle.scan_matcher
        )

        return new_particle


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
        meas_kernel_size: int,
        gaussian_sigma: float,
        proposal_alpha: float,
        proposal_beta: float,
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
                meas_kernel_size=meas_kernel_size,
                gaussian_sigma=gaussian_sigma,
                alpha=proposal_alpha,
                beta=proposal_beta,
            )

            # new_pose = corr_pose
            # p_weight = 1.0
            # prop_metrics = {
            #     "prop_mu": None,
            #     "prop_cov_matrix": None,
            #     "scan_match_pose": np.asarray(corr_pose, dtype=float),
            #     "pred_pose": np.asarray(pred_pose, dtype=float),
            #     "xjs": None,
            #     "xj_weights": None,
            #     "motion_probs": None,
            #     "meas_probs": None,
            # }
                

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
                # TODO: Adapt measurement model here
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

        # Pass through raw scan matcher diagnostics; evaluator owns metric computation.
        if prop_metrics is None:
            prop_metrics = {}
        prop_metrics["scan_matcher_info"] = particle.scan_matcher.get_info()

        return new_particle, scan_match_failed, scan_match_fallback_failed, prop_metrics


    def step(
        self,
        odom: Tuple[float, float],
        measurements_proposal: List[Tuple[float, float]],
        measurements_map_update: List[Tuple[float, float]],
        proposal_sigma_xy: float = 1.0,
        proposal_sigma_theta: float = 1.0,
        proposal_n_samples: int = 10,
        meas_kernel_size: int = 1,
        gaussian_sigma: float = 0.05,
        proposal_alpha: float = 0.5,
        proposal_beta: float = 2.0,
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
        # Init
        scan_match_failed_any = False
        scan_match_fallback_failed_any = False
        particle0_prop_metrics = None
        self._step_counter += 1
        step_idx = self._step_counter

        # Run initialization process of rbpf
        if self.init_status not in (InitStatus.SUCCESS, InitStatus.FAILED_ODOM_THRESHOLD):
            if self._stop_init_process(odom):
                dl, dr = odom
                self.init_status = InitStatus.FAILED_ODOM_THRESHOLD
                self.init_failure_reason = (
                    f"Initialization skipped at step {step_idx}: "
                    f"abs(dl)={abs(dl):.6f}, abs(dr)={abs(dr):.6f}, "
                    f"threshold={self.odom_threshold:.6f}"
                )
            elif self.init_counter < self.init_count_threshold:
                self.init_status = InitStatus.INITIALIZING

                # Run init process for each particle
                t_init_process = time.perf_counter()
                for i, p in enumerate(self.particles):
                    # Do init process
                    self.particles[i] = self.init_process(
                        particle=p,
                        measurements_map_update=measurements_map_update,
                    )
                # Measure time
                t_init_process_s = time.perf_counter() - t_init_process
                self._timing_stats_scan_match_only["t_init_process_sum_s"] += t_init_process_s
                self._timing_stats_scan_match_only["t_init_process_count"] += 1

                self.init_counter += 1

                if self.init_counter >= self.init_count_threshold:
                    self.init_status = InitStatus.SUCCESS

                # Keep step outputs structurally compatible during initialization
                # without generating pose metrics in map-only init mode.
                weights = np.array([p.weight for p in self.particles])
                norm = np.sum(weights)
                if norm == 0:
                    norm_weights = np.ones(len(weights)) / len(weights)
                else:
                    norm_weights = weights / norm

                for i in range(len(self.particles)):
                    self.particles[i].weight = norm_weights[i]

                neff = float(self.resampler.compute_neff(norm_weights))

                # Store step info
                self._last_step_info = {
                    "step": step_idx,
                    "mode": "initialization",
                    "init_status": self.init_status.value,
                    "init_counter": self.init_counter,
                    "init_count_threshold": self.init_count_threshold,
                    "odom_threshold": self.odom_threshold,
                    "init_failure_reason": self.init_failure_reason,
                    "neff": neff,
                    "scan_match_failed_any": None,
                    "scan_match_fallback_failed_any": None,
                    "best_particle_idx": None,
                    "best_particle_pose": None,
                    "best_particle_map": None,
                    "weighted_mean_pose": None,
                    "particle_weight_min": None,
                    "particle_weight_max": None,
                    "particle_weight_mean": None,
                    "timing_update_particles_s": None,
                    "timing_normalize_neff_s": None,
                    "timing_metrics_s": None,
                    "timing_resampling_s": None,
                    "proposal_metrics": None,
                    "resampled_indices": None,
                }

                return neff, self.particles[0].pose

        # Process each particle
        t_update_start = time.perf_counter()
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
                meas_kernel_size=meas_kernel_size,
                gaussian_sigma=gaussian_sigma,
                proposal_alpha=proposal_alpha,
                proposal_beta=proposal_beta,
            )
            scan_match_failed_any = scan_match_failed_any or scan_match_failed
            scan_match_fallback_failed_any = scan_match_fallback_failed_any or scan_match_fallback_failed
            if i == 0:
                particle0_prop_metrics = prop_metrics

        
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
            "scan_match_failed_any": scan_match_failed_any,
            "scan_match_fallback_failed_any": scan_match_fallback_failed_any,
            "best_particle_idx": best_idx,
            "best_particle_pose": best_particle_pose,
            "best_particle_map": self.particles[best_idx].scan_matcher.ogm.get_log_odds_map(),
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
            t_resampling_start = time.perf_counter()
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

            t_resampling_s = time.perf_counter() - t_resampling_start
            self._timing_stats["resampling_sum_s"] += t_resampling_s
            self._timing_stats["resampling_count"] += 1

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

        # Update pose by scan matcher
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

        # Extend map if necessary
        t_map_ext_start = time.perf_counter()
        extension_needed = True

        while extension_needed:
            extension_needed = particle.scan_matcher.ogm.map_extension_if_necessary(new_pose)

        t_map_ext_s = time.perf_counter() - t_map_ext_start
        self._timing_stats_scan_match_only["map_extension_sum_s"] += t_map_ext_s
        self._timing_stats_scan_match_only["map_extension_count"] += 1

        # Update ogm
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
        # Increase step counter
        self._step_counter += 1
        step_idx = self._step_counter

        # Initialization process
        if self.init_status not in (InitStatus.SUCCESS, InitStatus.FAILED_ODOM_THRESHOLD):
            if self._stop_init_process(odom):
                dl, dr = odom
                self.init_status = InitStatus.FAILED_ODOM_THRESHOLD
                self.init_failure_reason = (
                    f"Initialization skipped at step {step_idx}: "
                    f"abs(dl)={abs(dl):.6f}, abs(dr)={abs(dr):.6f}, "
                    f"threshold={self.odom_threshold:.6f}"
                )
            elif self.init_counter < self.init_count_threshold:
                self.init_status = InitStatus.INITIALIZING

                t_init_process = time.perf_counter()
                self.particles[0] = self.init_process(
                    particle=self.particles[0],
                    measurements_map_update=measurements_map_update,
                )
                t_init_process_s = time.perf_counter() - t_init_process
                self._timing_stats_scan_match_only["t_init_process_sum_s"] += t_init_process_s
                self._timing_stats_scan_match_only["t_init_process_count"] += 1

                self.init_counter += 1

                if self.init_counter >= self.init_count_threshold:
                    self.init_status = InitStatus.SUCCESS

                self._last_step_info_scan_match_only = {
                    "step": step_idx,
                    "mode": "initialization",
                    "init_status": self.init_status.value,
                    "init_counter": self.init_counter,
                    "init_count_threshold": self.init_count_threshold,
                    "odom_threshold": self.odom_threshold,
                    "init_failure_reason": self.init_failure_reason,
                    "scan_match_failed": False,
                    # "particle_pose": self.particles[0].pose,
                    "particle_map": self.particles[0].scan_matcher.ogm.get_log_odds_map(),
                    "timing_update_particle": None,
                    "timing_ogm_update": getattr(self, "_last_timing_sm_map_update_s", None),
                }

                # Keep compatibility for callers still using get_step_info().
                self._last_step_info = dict(self._last_step_info_scan_match_only)

                return 1.0, self.particles[0].pose
                    

        # Update particle based on scan matcher
        t_start_particle_update = time.perf_counter()
        self.particles[0], scan_match_failed = self.update_particle_scan_match_only(
            particle=self.particles[0],
            odom=odom,
            measurements_filter=measurements_filter,
            measurements_map_update=measurements_map_update,
        )
        t_update_particle_s = time.perf_counter() - t_start_particle_update
        self._timing_stats_scan_match_only["update_particle_sum_s"] += t_update_particle_s
        self._timing_stats_scan_match_only["update_particle_count"] += 1
    
        self._last_step_info_scan_match_only = {
            "step": step_idx,
            "mode": "scan_match_only",
            "init_status": self.init_status.value,
            "init_counter": self.init_counter,
            "init_count_threshold": self.init_count_threshold,
            "odom_threshold": self.odom_threshold,
            "init_failure_reason": self.init_failure_reason,
            "scan_match_failed": scan_match_failed,            
            "particle_pose": self.particles[0].pose,
            "particle_map": self.particles[0].scan_matcher.ogm.get_log_odds_map(),
            "timing_update_particle": t_update_particle_s,
            "timing_ogm_update": getattr(self, "_last_timing_sm_map_update_s", None),
        }

        # Keep compatibility for callers still using get_step_info().
        self._last_step_info = dict(self._last_step_info_scan_match_only)

        return 1.0, self.particles[0].pose

    
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
        meas_kernel_size: int,
        gaussian_sigma: float,
        proposal_alpha: float,
        proposal_beta: float,
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
                meas_kernel_size=meas_kernel_size,
                gaussian_sigma=gaussian_sigma,
                alpha=proposal_alpha,
                beta=proposal_beta,
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
        proposal_sigma_xy: float = 1.0,
        proposal_sigma_theta: float = 1.0,
        proposal_n_samples: int = 10,
        meas_kernel_size: int = 1,
        gaussian_sigma: float = 0.05,
        proposal_alpha: float = 0.5,
        proposal_beta: float = 2.0,
    ):
        # Init
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
                meas_kernel_size=meas_kernel_size,
                gaussian_sigma=gaussian_sigma,
                proposal_alpha=proposal_alpha,
                proposal_beta=proposal_beta,
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
            "scan_match_failed_any": scan_match_failed_any,
            "scan_match_fallback_failed_any": scan_match_fallback_failed_any,
            "best_particle_idx": best_idx,
            "best_particle_pose": best_particle_pose,
            "best_particle_map": self.particles[best_idx].scan_matcher.ogm.get_log_odds_map(),
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
        else: 
            self._last_step_info["timing_resampling_s"] = 0.0

        return neff, weighted_mean_pose

    
    def update_measurement_model_counters_fallback(self, result: dict):
        '''
        Update measurement model counters for proposal diagnostics. 
        '''
        self.meas_model_counters_fallback["call_count"] += 1
        self.meas_model_counters_fallback["valid_beam_count"] += result.get("valid_beam_count", 0)
        self.meas_model_counters_fallback["map_hit_count"] += result.get("map_hit_count", 0)
        self.meas_model_counters_fallback["no_map_hit_count"] += result.get("no_map_hit_count", 0)
        self.meas_model_counters_fallback["out_of_map_count"] += result.get("out_of_map_count", 0)
        self.meas_model_counters_fallback["unknown_ray_count"] += result.get("unknown_ray_count", 0)
        self.meas_model_counters_fallback["known_free_ray_count"] += result.get("known_free_ray_count", 0)
        self.meas_model_counters_fallback["unexpected_known_free_count"] += result.get("unexpected_known_free_count", 0)


    def update_particle_range_finder_model(
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
        cov_std_scale: float,
        cov_max_std_xy: float,
        cov_max_std_theta: float,
        min_std_xy: float,
        min_std_theta: float,
        proposal_rng: Optional[np.random.Generator] = None,
    ):
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

        # Compute proposal if scan matching was successful
        if corr_pose is not None:
            # Compute optimized proposal
            t_prop_start = time.perf_counter()
            
            new_pose, log_p_weight, prop_metrics = proposal.estimate_proposal_range_finder(
                scan_match_pose=corr_pose,
                particle=particle,
                odom=odom,
                measurements=measurements_proposal,
                motion_model=motion_model,
                measurement_model=measurement_model,
                sigma_xy=proposal_sigma_xy,
                sigma_theta=proposal_sigma_theta,
                n_samples=proposal_n_samples,
                cov_std_scale=cov_std_scale,
                cov_max_std_xy=cov_max_std_xy,
                cov_max_std_theta=cov_max_std_theta,
                min_std_xy=min_std_xy,
                min_std_theta=min_std_theta,
                rng=proposal_rng,
            )

            t_prop_s = time.perf_counter() - t_prop_start
            self._timing_stats["proposal_estimation_sum_s"] += t_prop_s
            self._timing_stats["proposal_estimation_count"] += 1

        # Fallback strategy if scan matching failed
        else:
            t_fallback_start = time.perf_counter()

            # Predict particle pose with motion model
            scan_match_failed = True

            new_pose = motion_model.predict_pose(
                pose=particle.pose,
                dl=dl,
                dr=dr,
            )

            # Fallback to Measurement model 
            results = measurement_model.likelihood(
                pose=new_pose,
                measurements=measurements_proposal,
                ogm=particle.scan_matcher.ogm,
            )

            self.update_measurement_model_counters_fallback(results)

            # Extract likilihood 
            log_p_weight = results.get("log_likelihood", FALLBACK_LOG_FLOOR)
            
            if not np.isfinite(log_p_weight):
                scan_match_fallback_failed = True
                log_p_weight = FALLBACK_LOG_FLOOR
            
            # Compute fallback time duration
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
            weight=particle.weight,
            scan_matcher=particle.scan_matcher,
        )

        # Update proposal metrics
        if prop_metrics is None:
            prop_metrics = {}
        prop_metrics["scan_matcher_info"] = particle.scan_matcher.get_info()
        # prop_metrics["measurement_model_counters_fallback"] = self.meas_model_counters_fallback

        return new_particle, log_p_weight, scan_match_failed, scan_match_fallback_failed, prop_metrics


    @staticmethod
    def normalize_weights(
        old_weights: List[float], 
        log_weight_increments: List[float],
    ) -> np.ndarray:
        '''
        Converts the log weight increments into probs and normalizes them. For safety the old weights are normalized first. 
        All results run through validation checks to avoid returning unexpected or wrong results.

        Parameters
        ----------
        old_weights : List[float]
            The old weights of the particles before the update step. Normalization takes place inside. 
        log_weight_increments : List[float]
            The log weight increments computed in the update step for each particle. The method assumes that they are 
            already finite (not Nan, nor inf). No check made inside here! Ensure valid values before!

        Returns
        -------
        np.ndarray
            The new normalized weights of the particles after applying the log weight increments.
        '''
        # Convert weights to numpy arrays for easier processing
        old_weights = np.asarray(old_weights, dtype=np.float64)
        log_weight_increments = np.asarray(log_weight_increments, dtype=np.float64)

        n = old_weights.shape[0]

        # Normalize old weights (safety)
        old_sum = np.sum(old_weights)

        # Validate old weights
        if old_sum <= 0.0 or not np.isfinite(old_sum):
            # Set all weights equal if normalizer of old weights is zero
            old_weights_normed = np.ones(n, dtype=np.float64) / n
        else:
            old_weights_normed = old_weights / old_sum

        # Compute new weights
        log_weights = np.empty(n, dtype=np.float64)
        # Compute log weights
        for i in range(n):
            old_w = max(old_weights_normed[i], 1e-300)
            log_weights[i] = np.log(old_w) + log_weight_increments[i]

        max_log_w = np.max(log_weights)

        # If all current updates failed or something exploded then keep old normalized weights
        if not np.isfinite(max_log_w):
            return old_weights_normed

        # Transform log weights to normal weights
        weights = np.exp(log_weights - max_log_w)
        weight_sum = np.sum(weights)

        # Check if new weights are valid, otherwise use old normalized weights
        if weight_sum <= 0.0 or not np.isfinite(weight_sum):
            return old_weights_normed

        return weights / weight_sum


    def resampling(
        self,
        norm_weights: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ):
        '''
        Resample particles if Neff falls below the threshold.

        Parameters
        ----------
        norm_weights : np.ndarray
            Normalized weights of the particles before resampling.
        '''
        # Compute neff from current normalized weights before resampling.
        neff = float(self.resampler.compute_neff(norm_weights))

        if neff < self.neff_threshold:
            # t_resampling_start = time.perf_counter()
            # Get inidices of particles that have survived
            indices = self.resampler.low_variance_sampler(
                norm_weights,
                rng=rng,
            )

            # Update particles
            new_partilces = []
            n_particles = len(self.particles)
            
            # Deep copy and update weight
            weight = 1.0 / n_particles
            for idx in indices:
                p = self.particles[idx].copy()

                p.weight = weight

                new_partilces.append(p)
        
            # Replace old particle set by new set
            self.particles = new_partilces       

            return indices
        return None 


    def step_range_finder_model(
        self,
        odom: Tuple[float, float],
        measurements_proposal: List[Tuple[float, float]],
        measurements_map_update: List[Tuple[float, float]],
        proposal_sigma_xy: float = 1.0,
        proposal_sigma_theta: float = 1.0,
        proposal_n_samples: int = 10,
        cov_std_scale: float = 0.5,
        cov_max_std_xy: float = 0.02,
        cov_max_std_theta: float = 0.02,
        min_std_xy: float = 0.0,
        min_std_theta: float = 0.0,
        run_seed: Optional[int] = None,
    ) -> None:
        '''
        Performs the update step of the particle filter for all particles. This includes the following steps:
        Steps:
            1. Initialize map if initialization is still active.
            2. Update each particle pose and map.
            3. Collect log weight increments.
            4. Normalize particle weights safely in log-space.
            5. Compute metrics before resampling.
            6. Resample if Neff falls below threshold.

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
        
        # Init measurement model counters
        self.meas_model_counters_fallback = {
            "call_count": 0,
            "valid_beam_count": 0,
            "map_hit_count": 0,
            "no_map_hit_count": 0,
            "out_of_map_count": 0,
            "unknown_ray_count": 0,
            "known_free_ray_count": 0,
            "unexpected_known_free_count": 0,
        }

        # Init time measurements (Currently not in use, but Init always with None!)
        t_update_particles_s = None
        t_norm_neff_s = None
        t_metrics_s = None
        t_resampling_s = None

        # Update step counter
        self._step_counter += 1
        step_idx = self._step_counter

        # Init Process (only first N iterations to get stable map points for scan matcher)
        scan_match_failed_any = False
        scan_match_fallback_failed_any = False
        best_p_prop_metrics = None

        # Run initialization process of rbpf
        if self.init_status not in (InitStatus.SUCCESS, InitStatus.FAILED_ODOM_THRESHOLD):
            if self._stop_init_process(odom):
                dl, dr = odom
                self.init_status = InitStatus.FAILED_ODOM_THRESHOLD
                self.init_failure_reason = (
                    f"Initialization skipped at step {step_idx}: "
                    f"abs(dl)={abs(dl):.6f}, abs(dr)={abs(dr):.6f}, "
                    f"threshold={self.odom_threshold:.6f}"
                )
            elif self.init_counter < self.init_count_threshold:
                self.init_status = InitStatus.INITIALIZING

                # Run init process for each particle
                t_init_process = time.perf_counter()
                for i, p in enumerate(self.particles):
                    # Do init process
                    self.particles[i] = self.init_process(
                        particle=p,
                        measurements_map_update=measurements_map_update,
                    )
                # Measure time
                t_init_process_s = time.perf_counter() - t_init_process
                self._timing_stats_scan_match_only["t_init_process_sum_s"] += t_init_process_s
                self._timing_stats_scan_match_only["t_init_process_count"] += 1

                self.init_counter += 1

                if self.init_counter >= self.init_count_threshold:
                    self.init_status = InitStatus.SUCCESS

                # TODO: Delete this properly. No weights needed in intialization as long as all have been initalized 
                # Keep step outputs structurally compatible during initialization
                # without generating pose metrics in map-only init mode.
                weights = np.array([p.weight for p in self.particles])
                norm = np.sum(weights)
                
                if norm == 0:
                    norm_weights = np.ones(len(weights)) / len(weights)
                else:
                    norm_weights = weights / norm

                for i in range(len(self.particles)):
                    self.particles[i].weight = norm_weights[i]

                neff = float(self.resampler.compute_neff(norm_weights))

                # Update rbpf info
                particle_poses = [p.pose for p in self.particles]
                particle_weights = [p.weight for p in self.particles]
                self._last_step_info = {
                    "mode": "initialization",
                    "step": step_idx,
                    "neff": neff,
                    "scan_match_failed_any": None,
                    "scan_match_fallback_failed_any": None,
                    "particle_poses_before_resampling": particle_poses,
                    "particle_weights_before_resampling": particle_weights,
                    "best_particle_idx": None,
                    "best_particle_pose": None,
                    "best_particle_map": None,
                    "weighted_mean_pose": None,
                    "particle_weight_min": None,
                    "particle_weight_max": None,
                    "particle_weight_mean": None,
                    "timing_update_particles_s": t_update_particles_s,
                    "timing_normalize_neff_s": t_norm_neff_s,
                    "timing_metrics_s": t_metrics_s,
                    "timing_resampling_s": t_resampling_s,
                    "proposal_metrics": best_p_prop_metrics,
                    "measurement_model_counters_fallback": None,
                    "particle_poses": particle_poses,
                    "particle_weights": particle_weights,
                    "init_status": self.init_status.value,
                    "init_counter": self.init_counter,
                    "init_count_threshold": self.init_count_threshold,
                    "init_failure_reason": self.init_failure_reason,
                    "odom_threshold": self.odom_threshold,
                    "resampled_indices": None
                }

                return None
        
        # Normal RBPF update
        # Process each particle
        old_weights = [p.weight for p in self.particles]
        prop_metrics_list = []
        log_particle_weights = []

        # Update particles
        self.particle_update_counter += 1
        for i, p in enumerate(self.particles):
            proposal_seed = _derive_operation_seed(
                run_seed=run_seed,
                step_idx=step_idx,
                stream_id=RNG_STREAM_PROPOSAL,
                particle_index=i,
            )
            proposal_rng = (
                np.random.default_rng(proposal_seed)
                if proposal_seed is not None
                else None
            )

            # Update particle
            (
                self.particles[i], 
                log_p_weight, 
                scan_match_failed, 
                scan_match_fallback_failed, 
                prop_metrics 
            ) = self.update_particle_range_finder_model(
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
                cov_std_scale=cov_std_scale,
                cov_max_std_xy=cov_max_std_xy,
                cov_max_std_theta=cov_max_std_theta,
                min_std_xy=min_std_xy,
                min_std_theta=min_std_theta,
                proposal_rng=proposal_rng,
            )

            log_particle_weights.append(log_p_weight)
            prop_metrics_list.append(prop_metrics)

            # Store info if scan matcher or its fallback failed
            scan_match_failed_any = scan_match_failed_any or scan_match_failed
            scan_match_fallback_failed_any = scan_match_fallback_failed_any or scan_match_fallback_failed
            
            # Use first particle metrics as proposal metrics
            # if i == 0:
            #     particle0_prop_metrics = prop_metrics

        # Normalize weights
        norm_weights = self.normalize_weights(
            old_weights=old_weights,
            log_weight_increments=log_particle_weights,
        )
        
        # Update weights
        for i in range(len(self.particles)):
            self.particles[i].weight = norm_weights[i]
                
        # Compute metrics        
        # neff
        neff = self.resampler.compute_neff(norm_weights)
        # Weighted mean pose before resampling.
        weighted_mean_pose = self._compute_weighted_mean_pose_from_particles(self.particles)
        # Get best particle pose before resampling
        best_idx = int(np.argmax(norm_weights))
        best_particle_pose = self.particles[best_idx].pose
        # Weight statistics before optional resampling.
        particle_weight_min = float(np.min(norm_weights))
        particle_weight_max = float(np.max(norm_weights))
        particle_weight_mean = float(np.mean(norm_weights))

        # Extract proposal metrics from best particle
        best_p_prop_metrics = prop_metrics_list[best_idx]    
        
        # Attention! particles might already be resampled so don't access self.particals for computing metrics!
        self._last_step_info = {
            "mode": "RBPF",
            "step": step_idx,
            "neff": neff,
            "scan_match_failed_any": scan_match_failed_any,
            "scan_match_fallback_failed_any": scan_match_fallback_failed_any,
            "particle_poses_before_resampling": [p.pose for p in self.particles],
            "particle_weights_before_resampling": norm_weights,
            "best_particle_idx": best_idx,
            "best_particle_pose": best_particle_pose,
            "best_particle_map": self.particles[best_idx].scan_matcher.ogm.get_log_odds_map(),
            "best_particle_map_meta": self.particles[best_idx].scan_matcher.ogm.get_map_meta(),
            "weighted_mean_pose": weighted_mean_pose,
            "particle_weight_min": particle_weight_min,
            "particle_weight_max": particle_weight_max,
            "particle_weight_mean": particle_weight_mean,
            "timing_update_particles_s": t_update_particles_s,
            "timing_normalize_neff_s": t_norm_neff_s,
            "timing_metrics_s": t_metrics_s,
            "timing_resampling_s": t_resampling_s,
            "proposal_metrics": best_p_prop_metrics,
            "measurement_model_counters_fallback": self.meas_model_counters_fallback,
        }

        # Resampling step
        resampling_seed = _derive_operation_seed(
            run_seed=run_seed,
            step_idx=step_idx,
            stream_id=RNG_STREAM_RESAMPLING,
        )
        resampling_rng = (
            np.random.default_rng(resampling_seed)
            if resampling_seed is not None
            else None
        )
        indices = self.resampling(
            norm_weights=norm_weights,
            rng=resampling_rng,
        )

        # Add step info after resampling
        self._last_step_info["particle_poses"] = [p.pose for p in self.particles]
        self._last_step_info["particle_weights"] = [p.weight for p in self.particles]
        self._last_step_info["resampled_indices"] = indices 

        return None


    def process_particle_update_results(
        self,
        results: List[ParticleUpdateResult]
    ) -> Tuple[np.ndarray, bool, bool]:
        # Init
        scan_match_failed_any = False
        scan_match_fallback_failed_any = False
        # best_p_prop_metrics = None   
        
        n_particles = len(self.particles)
        log_p_weights = np.full(n_particles, np.nan, dtype=float)

        # Process results
        for res in results:
            # Extract and validate index
            p_idx = int(res.particle_index)
            
            if p_idx is None or p_idx < 0 or p_idx >= n_particles:
                raise ValueError(f"Invalid particle index {p_idx} in ParticleUpdateResult.")
            
            # Update particle
            self.particles[p_idx] = res.updated_particle
            log_p_weights[p_idx] = res.log_p_weight

            scan_match_failed_any = scan_match_failed_any or res.scan_match_failed
            scan_match_fallback_failed_any = scan_match_fallback_failed_any or res.scan_match_fallback_failed

            # Accumulate measurement fallback counters
            if res.scan_match_failed is True:
                self.update_measurement_model_counters_fallback(res.meas_model_fallback_res)           

        return log_p_weights, scan_match_failed_any, scan_match_fallback_failed_any


    def step_parallel(
        self,
        particle_process_pool: ParticleProcessPool,
        odom: Tuple[float, float],
        measurements_proposal: List[Tuple[float, float]],
        measurements_map_update: List[Tuple[float, float]],
        proposal_sigma_xy: float = 1.0,
        proposal_sigma_theta: float = 1.0,
        proposal_n_samples: int = 10,
        cov_std_scale: float = 0.5,
        cov_max_std_xy: float = 0.02,
        cov_max_std_theta: float = 0.02,
        min_std_xy: float = 0.0,
        min_std_theta: float = 0.0,
        run_seed: Optional[int] = None,
    ) -> None:
        # Init measurement model counters
        self.meas_model_counters_fallback = {
            "call_count": 0,
            "valid_beam_count": 0,
            "map_hit_count": 0,
            "no_map_hit_count": 0,
            "out_of_map_count": 0,
            "unknown_ray_count": 0,
            "known_free_ray_count": 0,
            "unexpected_known_free_count": 0,
        }

        # Init time measurements (Currently not in use, but Init always with None!)
        t_update_particles_s = None
        t_norm_neff_s = None
        t_metrics_s = None
        t_resampling_s = None

        # Update step counter
        self._step_counter += 1
        step_idx = self._step_counter

        # Init Process (only first N iterations to get stable map points for scan matcher)
        scan_match_failed_any = False
        scan_match_fallback_failed_any = False
        best_p_prop_metrics = None

        # Run initialization process of rbpf
        if self.init_status not in (InitStatus.SUCCESS, InitStatus.FAILED_ODOM_THRESHOLD):
            if self._stop_init_process(odom):
                dl, dr = odom
                self.init_status = InitStatus.FAILED_ODOM_THRESHOLD
                self.init_failure_reason = (
                    f"Initialization skipped at step {step_idx}: "
                    f"abs(dl)={abs(dl):.6f}, abs(dr)={abs(dr):.6f}, "
                    f"threshold={self.odom_threshold:.6f}"
                )
            elif self.init_counter < self.init_count_threshold:
                self.init_status = InitStatus.INITIALIZING

                # Run init process for each particle
                t_init_process = time.perf_counter()
                for i, p in enumerate(self.particles):
                    # Do init process
                    self.particles[i] = self.init_process(
                        particle=p,
                        measurements_map_update=measurements_map_update,
                    )
                # Measure time
                t_init_process_s = time.perf_counter() - t_init_process
                self._timing_stats_scan_match_only["t_init_process_sum_s"] += t_init_process_s
                self._timing_stats_scan_match_only["t_init_process_count"] += 1

                self.init_counter += 1

                if self.init_counter >= self.init_count_threshold:
                    self.init_status = InitStatus.SUCCESS

                # TODO: Delete this properly. No weights needed in intialization as long as all have been initalized 
                # Keep step outputs structurally compatible during initialization
                # without generating pose metrics in map-only init mode.
                weights = np.array([p.weight for p in self.particles])
                norm = np.sum(weights)
                
                if norm == 0:
                    norm_weights = np.ones(len(weights)) / len(weights)
                else:
                    norm_weights = weights / norm

                for i in range(len(self.particles)):
                    self.particles[i].weight = norm_weights[i]

                neff = float(self.resampler.compute_neff(norm_weights))

                # Store step info
                # particle_poses = [p.pose for p in self.particles]
                # particle_weights = [p.weight for p in self.particles]
                # self._last_step_info = {
                #     "step": step_idx,
                #     "mode": "initialization",
                #     "init_status": self.init_status.value,
                #     "init_counter": self.init_counter,
                #     "init_count_threshold": self.init_count_threshold,
                #     "odom_threshold": self.odom_threshold,
                #     "init_failure_reason": self.init_failure_reason,
                #     "neff": neff,
                #     "scan_match_failed_any": None,
                #     "scan_match_fallback_failed_any": None,
                #     "best_particle_idx": None,
                #     "best_particle_pose": None,
                #     "best_particle_map": None,
                #     "weighted_mean_pose": None,
                #     "particle_weight_min": None,
                #     "particle_weight_max": None,
                #     "particle_weight_mean": None,
                #     "timing_update_particles_s": None,
                #     "timing_normalize_neff_s": None,
                #     "timing_metrics_s": None,
                #     "timing_resampling_s": None,
                #     "proposal_metrics": None,
                #     "measurement_model_counters_fallback": None
                # }

                # return neff, self.particles[0].pose

                # Update rbpf info
                particle_poses = [p.pose for p in self.particles]
                particle_weights = [p.weight for p in self.particles]
                self._last_step_info = {
                    "mode": "initialization",
                    "step": step_idx,
                    "neff": neff,
                    "scan_match_failed_any": None,
                    "scan_match_fallback_failed_any": None,
                    "particle_poses_before_resampling": particle_poses,
                    "particle_weights_before_resampling": particle_weights,
                    "best_particle_idx": None,
                    "best_particle_pose": None,
                    "best_particle_map": None,
                    "weighted_mean_pose": None,
                    "particle_weight_min": None,
                    "particle_weight_max": None,
                    "particle_weight_mean": None,
                    "timing_update_particles_s": t_update_particles_s,
                    "timing_normalize_neff_s": t_norm_neff_s,
                    "timing_metrics_s": t_metrics_s,
                    "timing_resampling_s": t_resampling_s,
                    "proposal_metrics": best_p_prop_metrics,
                    "measurement_model_counters_fallback": None,
                    "particle_poses": particle_poses,
                    "particle_weights": particle_weights,
                    "init_status": self.init_status.value,
                    "init_counter": self.init_counter,
                    "init_count_threshold": self.init_count_threshold,
                    "init_failure_reason": self.init_failure_reason,
                    "odom_threshold": self.odom_threshold,
                }
                
                return None
                
        
        # Normal RBPF update
        # Process each particle
        old_weights = [p.weight for p in self.particles]
        prop_metrics_list = []
        log_particle_weights = []

        # Update particles
        self.particle_update_counter += 1

        # Define particle tasks
        start_time_creating_tasks = time.perf_counter()
        tasks = []
        for particle_index, particle in enumerate(self.particles):
            task = ParticleUpdateTask(
                particle_index=particle_index,
                particle=particle,
                proposal_seed=_derive_operation_seed(
                    run_seed=run_seed,
                    step_idx=step_idx,
                    stream_id=RNG_STREAM_PROPOSAL,
                    particle_index=particle_index,
                ),

                motion_model=self.motion_model,
                measurement_model=self.measurement_model,   

                odom=odom,
                measurements_proposal=measurements_proposal,
                measurements_map_update=measurements_map_update,

                proposal_sigma_xy=proposal_sigma_xy,
                proposal_sigma_theta=proposal_sigma_theta,
                proposal_n_samples=proposal_n_samples,
                cov_std_scale=cov_std_scale,
                cov_max_std_xy=cov_max_std_xy,
                cov_max_std_theta=cov_max_std_theta,
                min_std_xy=min_std_xy,
                min_std_theta=min_std_theta,
                )
            tasks.append(task)
        
        duration_t_creating_task_ms = (time.perf_counter() - start_time_creating_tasks) * 1000.0
        
        # Update particles parallel
        start_time_updating_particles = time.perf_counter()
        results = particle_process_pool.map(
            worker_func=update_particle_worker,
            tasks=tasks
        )
        duration_t_updating_particles_ms = (time.perf_counter() - start_time_updating_particles) * 1000.0

        # Process results
        start_time_processing_results = time.perf_counter()
        (
            log_particle_weights, 
            scan_match_failed_any, 
            scan_match_fallback_failed_any, 
        ) = self.process_particle_update_results(results)

        duration_t_processing_results_ms = (time.perf_counter() - start_time_processing_results) * 1000.0

        # Normalize weights
        norm_weights = self.normalize_weights(
            old_weights=old_weights,
            log_weight_increments=log_particle_weights,
        )
        
        # Update weights
        for i in range(len(self.particles)):
            self.particles[i].weight = norm_weights[i]
         
        # Compute metrics        
        # neff
        neff = self.resampler.compute_neff(norm_weights)
        # Weighted mean pose before resampling.
        weighted_mean_pose = self._compute_weighted_mean_pose_from_particles(self.particles)

        # Get best particle pose before resampling
        best_idx = int(np.argmax(norm_weights))
        best_particle_pose = self.particles[best_idx].pose
        
        # Weight statistics before optional resampling.
        particle_weight_min = float(np.min(norm_weights))
        particle_weight_max = float(np.max(norm_weights))
        particle_weight_mean = float(np.mean(norm_weights))    

        # Extract proposal metrics from best particle
        best_p_prop_metrics = results[best_idx].prop_metrics 

        # Attention! particles might already be resampled so don't access self.particles for computing metrics!
        self._last_step_info = {
            "step": step_idx,
            "neff": neff,
            "scan_match_failed_any": scan_match_failed_any,
            "scan_match_fallback_failed_any": scan_match_fallback_failed_any,
            "particle_poses_before_resampling": [p.pose for p in self.particles],
            "particle_weights_before_resampling": norm_weights,
            "best_particle_idx": best_idx,
            "best_particle_pose": best_particle_pose,
            "best_particle_map": self.particles[best_idx].scan_matcher.ogm.get_log_odds_map(),
            "best_particle_map_meta": self.particles[best_idx].scan_matcher.ogm.get_map_meta(),
            "weighted_mean_pose": weighted_mean_pose,
            "particle_weight_min": particle_weight_min,
            "particle_weight_max": particle_weight_max,
            "particle_weight_mean": particle_weight_mean,
            "timing_update_particles_s": t_update_particles_s,
            "timing_normalize_neff_s": t_norm_neff_s,
            "timing_metrics_s": t_metrics_s,
            "timing_resampling_s": t_resampling_s,
            "proposal_metrics": best_p_prop_metrics,
            "measurement_model_counters_fallback": self.meas_model_counters_fallback,
        }

        # Resampling step
        resampling_seed = _derive_operation_seed(
            run_seed=run_seed,
            step_idx=step_idx,
            stream_id=RNG_STREAM_RESAMPLING,
        )
        resampling_rng = (
            np.random.default_rng(resampling_seed)
            if resampling_seed is not None
            else None
        )
        indices = self.resampling(
            norm_weights=norm_weights,
            rng=resampling_rng,
        )

        # Add step info after resampling
        self._last_step_info["particle_poses"] = [p.pose for p in self.particles]
        self._last_step_info["particle_weights"] = [p.weight for p in self.particles]
        self._last_step_info["resampled_indices"] = indices

        # print(f"\nTime for parallel particle update:") 
        # print(f"  creating tasks: {duration_t_creating_task_ms:.6f} ms")
        # print(f"  updating particles: {duration_t_updating_particles_ms:.6f} ms")
        # print(f"  processing results: {duration_t_processing_results_ms:.6f} ms")

        return neff, weighted_mean_pose
