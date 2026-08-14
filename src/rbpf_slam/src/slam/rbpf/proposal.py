from typing import List, Tuple, Dict, Optional
import time

import numpy as np

from sklearn.neighbors import NearestNeighbors

from slam.infrastructure.defs import Pose2D
from slam.rbpf.particle import Particle
from slam.rbpf.motion_model import MotionModel
from slam.rbpf.measurement_model import MeasurementModel


# Fallback log-likelihood value when the likelihood computation fails or returns invalid values.
# Penalty if log likelihood is invalid
INVALID_LOG_LIKELIHOOD = -1.0e12
# FALLBACK_LOG_FLOOR = -80.0


class ProposalEstimator:
    '''
    Class for estiating/approximating the proposal distribution by a gaussian distribution. The estimation is done
    based on initial pose, odometry and measurements. 
    '''
    IDX_x=0
    IDX_y=1
    IDX_THETA=2

    def __init__(self):
        self.t_sample_poses = 0.0
        self.t_pred_poses = 0.0
        self.t_motion_model = 0.0
        self.t_meas_model = 0.0
        self.t_compute_prop_params = 0.0
        self.t_sample_from_prop = 0.0


    @staticmethod
    def _compute_mean_time(time: float, count: Optional[int] = None):
        '''
        Compute the mean time given the total time and count.
        '''
        if time is None or time <= 0.0 or count is None or count <= 0:
            return None
        return time / count


    @staticmethod
    def _filter_time(time: float):
        '''
        Filter out invalid or negative timing values. 
        '''
        if time is None or time <= 0.0:
            return None
        return time


    def evaluate_timings(self):
        '''
        Evaluate and filter the timing metrics for proposal estimation. 
        '''
        self.t_sample_poses = self._filter_time(self.t_sample_poses)
        self.t_pred_poses = self._filter_time(self.t_pred_poses)
        self.t_motion_model = self._filter_time(self.t_motion_model)
        self.t_meas_model = self._filter_time(self.t_meas_model)
        self.t_compute_prop_params = self._filter_time(self.t_compute_prop_params)
        self.t_sample_from_prop = self._filter_time(self.t_sample_from_prop)
        

    def sample_poses(self, pose: Pose2D, sigma_xy: float, sigma_theta: float, n_samples: int) -> np.ndarray:
        '''
        Defines a quadratic sample area around the given pose of size sigma_xy and sigma_theta. The number of samples we draw
        per dof (x,y,theta) is defined by n_samples.
        For each position xy we draw n_samples times theta from a Gaussian distribution.
        The total number of samples = n_samples**2 * n_samples = n_samples**3
            -> n_samples in x
            -> n_samples in y
            -> n_samples in theta per x,y

        Parameters
        ----------
        pose: Pose2D
            The pose that marks the center of the sampling area.
        sigma_xy: float
            The standard deviation in x,y-direction used for the sampling area.
        sigma_theta: float
            The standard deviation in theta-direction used for the sampling area.
        n_samples: int
            The number of samples to draw.
        
        Returns
        -------
        samples: np.ndarray
            The sampled poses around the given pose where each sample is a x, y, theta pose.
        '''
        # Decompose the pose
        x, y, theta = pose

        # Draw samples from Gaussian distributions for x, y, and theta
        samples = np.zeros((n_samples, 3))
        samples[:, self.IDX_x] = np.random.normal(x, sigma_xy, n_samples)
        samples[:, self.IDX_y] = np.random.normal(y, sigma_xy, n_samples)
        angles = np.random.normal(theta, sigma_theta, n_samples)

        # Ensure angles are normalized to the range [-pi, pi]
        samples[:, self.IDX_THETA] = np.arctan2(np.sin(angles), np.cos(angles))

        return samples


    @staticmethod
    def sample_poses_deterministic(
        pose: Pose2D,
        sigma_xy: float, 
        sigma_theta: float,
        n_samples_dir: int = 3
    ) -> Tuple[np.ndarray, int]:
        '''
        Defines a quadratic, deterministic, sampling area around the given pose of size sigma_xy and sigma_theta. The number of 
        samples we draw per dof (x,y,theta) is defined by n_samples_dir.
        We draw n_samples_dir^3 samples in total. The samples are deterministically drawn from a grid around the given pose.
        sigma_xy defines the range in x,y-direction and sigma_theta defines the range in theta-direction.
        

        The total number of samples = n_samples_dir**2 * n_samples_dir = n_samples_dir**3
            -> n_samples_dir in x
            -> n_samples_dir in y
            -> n_samples_dir in theta per x,y

        Parameters
        ----------
        pose: Pose2D
            The pose that marks the center of the sampling area.
        sigma_xy: float
            The standard deviation in x,y-direction used for the sampling area.
        sigma_theta: float
            The standard deviation in theta-direction used for the sampling area.
        n_samples_dir: int
            The number of samples to draw per degree of freedom (x, y, theta).
        
        Returns
        -------
        samples: np.ndarray
            The sampled poses around the given pose where each sample is a x, y, theta pose.
        '''
        # List to store xj
        n_xj = n_samples_dir**3
        samples = np.zeros((n_xj, 3))
        
        x, y, theta = pose

        xs = np.linspace(
            x - sigma_xy,
            x + sigma_xy,
            n_samples_dir
        )

        ys = np.linspace(
            y - sigma_xy,
            y + sigma_xy,
            n_samples_dir,
        )

        thetas = np.linspace(
            theta - sigma_theta,
            theta + sigma_theta,
            n_samples_dir,
        )

        # Store samples
        k = 0
        for sx in xs:
            for sy in ys:
                for st in thetas:
                    samples[k, :] = [sx, sy, st]
                    k += 1

        return samples, n_xj


    def update_measurement_model_counters(self, result: dict):
        '''
        Update measurement model counters for proposal diagnostics. 
        '''
        self.meas_model_counters["call_count"] += 1
        self.meas_model_counters["valid_beam_count"] += result.get("valid_beam_count", 0)
        self.meas_model_counters["map_hit_count"] += result.get("map_hit_count", 0)
        self.meas_model_counters["no_map_hit_count"] += result.get("no_map_hit_count", 0)
        self.meas_model_counters["out_of_map_count"] += result.get("out_of_map_count", 0)
        self.meas_model_counters["unknown_ray_count"] += result.get("unknown_ray_count", 0)
        self.meas_model_counters["known_free_ray_count"] += result.get("known_free_ray_count", 0)
        self.meas_model_counters["unexpected_known_free_count"] += result.get("unexpected_known_free_count", 0)
        

    def _compute_proposal_params(
        self,
        scan_match_pose: Pose2D,
        particle: Particle,
        odom: Tuple[float, float],
        measurements: List[Tuple[float, float]],
        motion_model: MotionModel,
        measurement_model: MeasurementModel,
        sigma_xy: float=1.0,
        sigma_theta: float=1.0,
        n_samples: int=3,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Pose2D, float]:
        '''
        Estimates the parameters of the proposal distribution (approximated by a Gaussian distribution) based on the
        scan match pose, odometry, and measurements. 

        Parameters
        ----------
        scan_match_pose: Pose2D
            The pose obtained from scan matching.
        particle: Particle
            The particle for which the proposal is being estimated.
        odom: Tuple[float, float]
            The odometry readings (dl, dr) for the particle.
        measurements: List[Tuple[float, float]]
            The measurements (e.g., laser scans) associated with the particle.
        motion_model: MotionModel
            The motion model used to predict the particle's pose.
        measurement_model: MeasurementModel
            The measurement model used to compute the likelihood of the measurements.
        sigma_xy: float
            The standard deviation in x,y-direction used for the sampling area.
        sigma_theta: float
            The standard deviation in theta-direction used for the sampling area.
        n_samples: int
            The number of samples to draw for estimating the proposal distribution.

        Returns
        -------
        mu: np.ndarray
            The mean of the proposal distribution.
        cov: np.ndarray
            The covariance matrix of the proposal distribution.
        norm: float
            The normalization constant for the proposal distribution.
        samples: np.ndarray
            The sampled poses used for estimating the proposal distribution.
        weights: np.ndarray
            The weights that correspond to the sampled poses.
        meas_probs: np.ndarray
            The measurement probabilities for each sampled pose.
        motion_probs: np.ndarray
            The motion probabilities for each sampled pose.
        pred_pose: Pose2D
            The predicted pose based on odometry.
        log_eta: float
            The log measurement likelihood of the given particle.
        '''
        # Define vars
        norm = 0.0
        mu = np.zeros(3)

        # Reset measurement model counters for diagnostics
        self.meas_model_counters = {
            "call_count": 0,
            "valid_beam_count": 0,
            "map_hit_count": 0,
            "no_map_hit_count": 0,
            "out_of_map_count": 0,
            "unknown_ray_count": 0,
            "known_free_ray_count": 0,
            "unexpected_known_free_count": 0,
        }

        t_sample_poses_start = time.perf_counter()
        samples, n_samples = self.sample_poses_deterministic(
            pose=scan_match_pose,
            sigma_xy=sigma_xy,
            sigma_theta=sigma_theta,
            n_samples_dir=n_samples,
        )
        # Compute sample poses time
        self.t_sample_poses = time.perf_counter() - t_sample_poses_start

        # Predict particle pose based on odometry and old particle pose 
        t_pred_pose_start = time.perf_counter()
        dl, dr = odom
        pred_pose = motion_model.predict_pose(
            pose=particle.pose,
            dl=dl,
            dr=dr,
        )
        # Compute pose prediction time
        self.t_pred_poses = time.perf_counter() - t_pred_pose_start

        # Check if measurements contain nan values
        # if np.isnan(measurements).any():
        #     print("\nProposal: Measruement model contains nan value")

        # Compute measurement model likelihoods for each sample  
        t_meas_model_start = time.perf_counter()

        # Compute measurement model likelihoods
        meas_model_results = measurement_model.likelihood_batch(
            poses=samples,
            measurements=measurements,
            ogm=particle.scan_matcher.ogm
        )

        # Extract results
        # Extract likelihood
        log_likelihoods = meas_model_results[0]
        mean_abs_errors = meas_model_results[1]

        # Extract counters
        self.meas_model_counters["call_count"] = meas_model_results[2]
        self.meas_model_counters["valid_beam_count"] = meas_model_results[3]
        self.meas_model_counters["map_hit_count"] = meas_model_results[4]
        self.meas_model_counters["no_map_hit_count"] = meas_model_results[5]
        self.meas_model_counters["out_of_map_count"] = meas_model_results[6]
        self.meas_model_counters["unknown_ray_count"] = meas_model_results[7]
        self.meas_model_counters["known_free_ray_count"] = meas_model_results[8]
        self.meas_model_counters["unexpected_known_free_count"] = meas_model_results[9]
                    
        # Convert log-likelihoods to probabilities in save manner
        max_log_likelihood = np.max(log_likelihoods)

        # Compute measurement model time
        self.t_meas_model = time.perf_counter() - t_meas_model_start

        # Ensure valid log-likelihoods and compute measurement probabilities
        if not np.isfinite(max_log_likelihood):
            mu = np.asarray(scan_match_pose, dtype=float)
            cov = 1e-6 * np.eye(3)
            weights = np.ones(samples.shape[0], dtype=float)
            meas_probs = np.ones(samples.shape[0], dtype=float)
            motion_probs = np.ones(samples.shape[0], dtype=float)
            norm =  1e-12
            log_eta = INVALID_LOG_LIKELIHOOD
            return mu, cov, norm, samples, weights, meas_probs, motion_probs, pred_pose, log_eta
        else:
            meas_probs = np.exp(log_likelihoods - max_log_likelihood)
                
        # Compute motion probs
        t_motion_model_start = time.perf_counter()
        motion_probs = motion_model.motion_probability_batch(
            x_new=samples,
            x_prev=pred_pose,
        )
        self.t_motion_model = time.perf_counter() - t_motion_model_start

        # Compute proposal params
        t_compute_proposal_start = time.perf_counter()

        # Compute weights 
        weights = motion_probs * meas_probs
        norm = np.sum(weights)                       

        if (not np.isfinite(norm)) or norm <= 1e-12:
            # Fallback when all sample weights collapse to zero/invalid values.
            mu = np.asarray(scan_match_pose, dtype=float)
            cov = 1e-6 * np.eye(3)
            weights = np.ones(samples.shape[0], dtype=float)
            meas_probs = np.ones(samples.shape[0], dtype=float)
            motion_probs = np.ones(samples.shape[0], dtype=float)
            norm =  1e-12
            log_eta = INVALID_LOG_LIKELIHOOD
            return mu, cov, norm, samples, weights, meas_probs, motion_probs, pred_pose, log_eta
        
        # Compute log eta for success case
        log_eta = max_log_likelihood + np.log(norm)

        # Vectorize computation of mu and cov
        # Compute mu
        mu = np.sum(samples * weights[:, None], axis=0) / norm

        # TODO: Add this later on again when tuning from beginning
        # Handle angles correctly
        # mu[self.IDX_THETA] = np.arctan2(
        #     np.sum(weights * np.sin(samples[:, self.IDX_THETA])),
        #     np.sum(weights * np.cos(samples[:, self.IDX_THETA]))
        # )

        # Compute covariance matrix
        # Compute deviation from the mean
        x_minus_mu = samples - mu
        # Ensure valid angles
        x_minus_mu[:, self.IDX_THETA] = np.arctan2(np.sin(x_minus_mu[:, self.IDX_THETA]), np.cos(x_minus_mu[:, self.IDX_THETA]))
        # Compute noralized covariance
        cov = (weights[:, None] * x_minus_mu).T @ x_minus_mu / norm

        # Ensure covariance matrix is positive definite by adding small values to diagonal
        cov += 1e-6 * np.eye(3)

        self.t_compute_prop_params = time.perf_counter() - t_compute_proposal_start

        return mu, cov, norm, samples, weights, meas_probs, motion_probs, pred_pose, log_eta
    

    @staticmethod
    def __shrink_and_limit_cov(
        cov: np.ndarray,
        std_scale: float=0.5,
        max_std_xy: Optional[float]=None,
        min_std_xy: Optional[float]=None,
        max_std_theta: Optional[float]=None,
        min_std_theta: Optional[float]=None
    ):
        '''
        Shrinks the given covariance matrix based on the given scale parameter, while ensuring that the resulting standard deviations
        are within the specified limits. Returns the modified covariance matrix.
        '''
        cov = np.asarray(cov, dtype=float)
        cov = 0.5 * (cov + cov.T)

        old_var = np.maximum(np.diag(cov), 1e-12)
        old_std = np.sqrt(old_var)

        new_std = old_std * std_scale

        new_std[0] = np.clip(new_std[0], min_std_xy, max_std_xy)
        new_std[1] = np.clip(new_std[1], min_std_xy, max_std_xy)
        new_std[2] = np.clip(new_std[2], min_std_theta, max_std_theta)

        scale = new_std / old_std
        scale_matrix = np.diag(scale)

        new_cov = scale_matrix @ cov @ scale_matrix
        new_cov = 0.5 * (new_cov + new_cov.T)

        new_cov += 1e-9 * np.eye(3)

        return new_cov


    def _sample_from_proposal_limit(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        std_scale: float=0.5,
        max_std_xy: Optional[float]=None,
        min_std_xy: Optional[float]=None,
        max_std_theta: Optional[float]=None,
        min_std_theta: Optional[float]=None,
        rng: Optional[np.random.Generator]=None,
    ):
        '''
        Shrinks the given cov based in the given scale param, while ensuring the limits. Returns the pose sampled from that
        proposal with the shrunk and limited covariance.
        '''
        new_cov = self.__shrink_and_limit_cov(
            cov=cov,
            std_scale=std_scale,
            max_std_xy=max_std_xy,
            min_std_xy=min_std_xy,
            max_std_theta=max_std_theta,
            min_std_theta=min_std_theta
        )

        if rng is None:
            new_pose = np.random.multivariate_normal(mean=mu, cov=new_cov)
        else:
            new_pose = rng.multivariate_normal(mean=mu, cov=new_cov)
        new_pose[self.IDX_THETA] = np.arctan2(np.sin(new_pose[self.IDX_THETA]), np.cos(new_pose[self.IDX_THETA]))

        return new_pose


    def estimate_proposal(
        self,
        scan_match_pose: Pose2D,
        particle: Particle,
        odom: Tuple[float, float],
        measurements: List[Tuple[float, float]],
        motion_model: MotionModel,
        measurement_model: MeasurementModel,
        sigma_xy: float=1.0,
        sigma_theta: float=1.0,
        n_samples: int=3,
        cov_std_scale: float=0.5,
        cov_max_std_xy: float=0.02,
        cov_max_std_theta: float=0.02,
        min_std_xy: float=0.0,
        min_std_theta: float=0.0,
        rng: Optional[np.random.Generator]=None,
    ):
        '''
        Estimates the new pose of the given particle by sampling from the estimated proposal distribution. The estimated proposal
        distribution is approximated by a Gaussian distribution with mean and covariance. Its parameters are computed based on
        the scan match pose, odometry, and measurements. 

        Parameters
        ----------
        scan_match_pose: Pose2D
            The pose obtained from scan matching.
        particle: Particle
            The particle for which the proposal is being estimated.
        odom: Tuple[float, float]
            The odometry readings (dl, dr) for the particle.
        measurements: List[Tuple[float, float]]
            The measurements (e.g., laser scans) associated with the particle.
        motion_model: MotionModel
            The motion model used to predict the particle's pose.
        measurement_model: MeasurementModel
            The measurement model used to compute the likelihood of the measurements.
        sigma_xy: float
            The standard deviation in x,y-direction used for the sampling area.
        sigma_theta: float
            The standard deviation in theta-direction used for the sampling area.
        n_samples: int
            The number of samples to draw for estimating the proposal distribution.
        cov_std_scale: float
            Scaling factor to downscale the var/cov of the cov matrix.
        cov_max_std_xy: float
            Maximum standard deviation in x,y-direction for the proposal distribution.
        cov_max_std_theta: float
            Maximum standard deviation in theta-direction for the proposal distribution.
        min_std_xy: float
            Minimum standard deviation in x,y-direction for the proposal distribution.
        min_std_theta: float
            Minimum standard deviation in theta-direction for the proposal distribution.
        rng: Optional[np.random.Generator]
            Optional random number generator for reproducibility.  

        Returns
        -------
        new_p_pose: Pose2D
            The newly estimated pose for the particle.
        log_eta: float
            The log measurement likelihood of the given particle.
        info: Dict[str, Any]
            Dictionary containing additional information about the proposal estimation, including:
                - "prop_mu": The mean of the proposal distribution.
                - "prop_cov_matrix": The covariance matrix of the proposal distribution.
                - "scan_match_pose": The scan match pose used for estimation.
                - "pred_pose": The predicted pose based on odometry.
                - "xjs": The sampled poses used for estimating the proposal distribution.
                - "xj_weights": The weights that correspond to the sampled poses.
                - "motion_probs": The motion probabilities for each sampled pose.
                - "meas_probs": The measurement probabilities for each sampled pose.
                - "measurement_model_counters": Counters related to the measurement model evaluation.
                - "prop_timings": Timings related to the proposal estimation.
        '''
        # Rest proposal timings
        self.t_sample_poses = 0.0
        self.t_pred_poses = 0.0
        self.t_motion_model = 0.0
        self.t_meas_model = 0.0
        self.t_compute_prop_params = 0.0
        self.t_sample_from_prop = 0.0

        # Compute proposal parameter with ray tracing 
        mu, cov, norm, xjs, xj_weights, meas_probs, motion_probs, pred_pose, log_eta = self._compute_proposal_params(
            scan_match_pose=scan_match_pose,
            particle=particle,
            odom=odom,
            measurements=measurements,
            motion_model=motion_model,
            measurement_model=measurement_model,
            sigma_xy=sigma_xy,
            sigma_theta=sigma_theta,
            n_samples=n_samples,
        )

        # Estimate new particle pose
        # TODO: Repalce that at the end to get different poses for teh particles. THink about how to do/sample 
        t_sample_from_proposal_start = time.perf_counter()

        # Estimate new particle pose
        # new_p_pose = mu
        new_p_pose = self._sample_from_proposal_limit(
            mu=mu,
            cov=cov,
            std_scale=cov_std_scale,
            max_std_xy=cov_max_std_xy,
            min_std_xy=min_std_xy,
            max_std_theta=cov_max_std_theta,
            min_std_theta=min_std_theta,
            rng=rng,
        )


        self.t_sample_from_prop = time.perf_counter() - t_sample_from_proposal_start

        prop_timings = {
            "t_sample_poses": self.t_sample_poses,
            "t_pred_poses": self.t_pred_poses,
            "t_motion_model": self.t_motion_model,
            "t_meas_model": self.t_meas_model,
            "t_compute_prop_params": self.t_compute_prop_params,
            "t_sample_from_prop": self.t_sample_from_prop,
        }

        info = {
            "prop_mu": mu,
            "prop_cov_matrix": cov,
            "scan_match_pose": np.asarray(scan_match_pose, dtype=float),
            "pred_pose": np.asarray(pred_pose, dtype=float),
            "xjs": xjs,
            "xj_weights": xj_weights,
            "motion_probs": motion_probs,
            "meas_probs": meas_probs,
            "measurement_model_counters": self.meas_model_counters,
            "prop_timings": prop_timings,
        }

        return new_p_pose, log_eta, info


    def estimate_proposal_one_particle(
        self,
        scan_match_pose: Pose2D,
        particle: Particle,
        odom: Tuple[float, float],
        measurements: List[Tuple[float, float]],
        motion_model: MotionModel,
        measurement_model: MeasurementModel,
        sigma_xy: float=1.0,
        sigma_theta: float=1.0,
        n_samples: int=3,
    ):
        '''
        Approximates the proposal distribution by a Gaussian distribution with mean and covariance. Its 
        parameters are computed based on the scan match pose, odometry, and measurements. The new particle
        pose will be the mean if the proposal distribution. 

        scan_match_pose: Pose2D
            The pose obtained from scan matching.
        particle: Particle
            The particle for which the proposal is being estimated.
        odom: Tuple[float, float]
            The odometry readings (dl, dr) for the particle.
        measurements: List[Tuple[float, float]]
            The measurements (e.g., laser scans) associated with the particle.
        motion_model: MotionModel
            The motion model used to predict the particle's pose.
        measurement_model: MeasurementModel
            The measurement model used to compute the likelihood of the measurements.
        sigma_xy: float
            The standard deviation in x,y-direction used for the sampling area.
        sigma_theta: float
            The standard deviation in theta-direction used for the sampling area.
        n_samples: int
            The number of samples to draw for estimating the proposal distribution.

        Returns
        -------
        new_p_pose: Pose2D
            The newly estimated pose for the particle.
        log_eta: float
            The log measurement likelihood of the given particle.
        info: Dict[str, Any]
            Dictionary containing additional information about the proposal estimation, including:
                - "prop_mu": The mean of the proposal distribution.
                - "prop_cov_matrix": The covariance matrix of the proposal distribution.
                - "scan_match_pose": The scan match pose used for estimation.
                - "pred_pose": The predicted pose based on odometry.
                - "xjs": The sampled poses used for estimating the proposal distribution.
                - "xj_weights": The weights that correspond to the sampled poses.
                - "motion_probs": The motion probabilities for each sampled pose.
                - "meas_probs": The measurement probabilities for each sampled pose.
                - "measurement_model_counters": Counters related to the measurement model evaluation.
                - "prop_timings": Timings related to the proposal estimation.
        '''
        # Rest proposal timings
        self.t_sample_poses = 0.0
        self.t_pred_poses = 0.0
        self.t_motion_model = 0.0
        self.t_meas_model = 0.0
        self.t_compute_prop_params = 0.0
        self.t_sample_from_prop = 0.0

        # Compute proposal parameter with ray tracing 
        mu, cov, norm, xjs, xj_weights, meas_probs, motion_probs, pred_pose, log_eta = self._compute_proposal_params(
            scan_match_pose=scan_match_pose,
            particle=particle,
            odom=odom,
            measurements=measurements,
            motion_model=motion_model,
            measurement_model=measurement_model,
            sigma_xy=sigma_xy,
            sigma_theta=sigma_theta,
            n_samples=n_samples,
        )

        # Estimate new particle pose
        t_sample_from_proposal_start = time.perf_counter()

        # Estimate new particle pose
        new_p_pose = mu
        
        self.t_sample_from_prop = time.perf_counter() - t_sample_from_proposal_start

        prop_timings = {
            "t_sample_poses": self.t_sample_poses,
            "t_pred_poses": self.t_pred_poses,
            "t_motion_model": self.t_motion_model,
            "t_meas_model": self.t_meas_model,
            "t_compute_prop_params": self.t_compute_prop_params,
            "t_sample_from_prop": self.t_sample_from_prop,
        }

        info = {
            "prop_mu": mu,
            "prop_cov_matrix": cov,
            "scan_match_pose": np.asarray(scan_match_pose, dtype=float),
            "pred_pose": np.asarray(pred_pose, dtype=float),
            "xjs": xjs,
            "xj_weights": xj_weights,
            "motion_probs": motion_probs,
            "meas_probs": meas_probs,
            "measurement_model_counters": self.meas_model_counters,
            "prop_timings": prop_timings,
        }

        return new_p_pose, log_eta, info
