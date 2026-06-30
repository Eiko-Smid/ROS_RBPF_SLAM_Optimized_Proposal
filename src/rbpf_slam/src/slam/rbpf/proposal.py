from typing import List, Tuple, Dict, Optional
import time

import numpy as np

from sklearn.neighbors import NearestNeighbors

from slam.infrastructure.defs import Pose2D
from slam.rbpf.particle import Particle
from slam.rbpf.motion_model import MotionModel
from slam.rbpf.measurement_model import MeasurementModel


INVALID_LOG_LIKELIHOOD = -1.0e12
FALLBACK_LOG_FLOOR = -80.0


class ProposalEstimator:
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
        x, y, theta = pose

        samples = np.zeros((n_samples, 3))
        samples[:, self.IDX_x] = np.random.normal(x, sigma_xy, n_samples)
        samples[:, self.IDX_y] = np.random.normal(y, sigma_xy, n_samples)
        angles = np.random.normal(theta, sigma_theta, n_samples)
        samples[:, self.IDX_THETA] = np.arctan2(np.sin(angles), np.cos(angles))

        return samples


    @staticmethod
    def sample_poses_deterministic(pose: Pose2D, sigma_xy: float, sigma_theta: float, n_samples_dir: int = 3) -> Tuple[np.ndarray, int]:
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


    def compute_proposal_param(
        self,
        scan_match_pose: Pose2D,
        particle: Particle,
        odom: Tuple[float, float],
        measurements: List[Tuple[float, float]],
        neighbor: NearestNeighbors,
        motion_model: MotionModel,
        measurement_model: MeasurementModel,
        sigma_xy: float=1.0,
        sigma_theta: float=1.0,
        n_samples: int=3,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        # Define vars
        norm = 0.0
        mu = np.zeros(3)

        # Sample k new poses around scan matcher pose
        # samples = self.sample_poses(
        #     pose=scan_match_pose,
        #     sigma_xy=sigma_xy,
        #     sigma_theta=sigma_theta,
        #     n_samples=n_samples,
        # )

        samples, n_samples = self.sample_poses_deterministic(
            pose=scan_match_pose,
            sigma_xy=sigma_xy,
            sigma_theta=sigma_theta,
            n_samples_dir=n_samples,
        )

        weights = np.zeros(shape=(n_samples))

        # Predict particle pose based on odometry and old particle pose 
        dl, dr = odom
        pred_pose = motion_model.predict_pose(
            pose=particle.pose,
            dl=dl,
            dr=dr,
        )

        # Compute weights and normalizer
        for i in range(samples.shape[0]):
            xj = samples[i, :]
            meas_prob = measurement_model.likelihood(
                pose=xj,
                measurements=measurements,
                scan_matcher= particle.scan_matcher,
                neighbor=neighbor,
            )
            
            motion_prob = motion_model.motion_probability(
                x_new=xj,
                x_prev=pred_pose,
            )

            # Compute probability and add to normalizer 
            w = meas_prob * motion_prob
            weights[i] = w            

        # Vectorized computation of mu and cov
        norm = np.sum(weights)

        if (not np.isfinite(norm)) or norm <= 1e-12:
            # Fallback when all sample weights collapse to zero/invalid values.
            mu = np.asarray(scan_match_pose, dtype=float)
            cov = 1e-6 * np.eye(3)
            return mu, cov, 1e-12, np.ones(samples.shape[0], dtype=float), pred_pose

        # Compute mu
        mu = np.sum(samples * weights[:, None], axis=0) / norm

        # Compute covariance matrix
        # Compute deviation from the mean
        x_minus_mu = samples - mu
        # Ensure valid angles
        x_minus_mu[:, self.IDX_THETA] = np.arctan2(np.sin(x_minus_mu[:, self.IDX_THETA]), np.cos(x_minus_mu[:, self.IDX_THETA]))
        # Compute noralized covariance
        cov = (weights[:, None] * x_minus_mu).T @ x_minus_mu / norm

        # Ensure covariance matrix is positive definite by adding small values to diagonal
        cov += 1e-6 * np.eye(3)

        return mu, cov, norm


    def compute_proposal_param_gmapping(
        self,
        scan_match_pose: Pose2D,
        particle: Particle,
        odom: Tuple[float, float],
        measurements: List[Tuple[float, float]],
        motion_model: MotionModel,
        measurement_model: MeasurementModel,
        meas_kernel_size: int=1,
        gaussian_sigma: float=0.05,
        sigma_xy: float=1.0,
        sigma_theta: float=1.0,
        n_samples_dir: int=3,
        alpha: float=0.5,
        beta: float=2.0,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Pose2D]:
        # Define vars
        norm = 0.0
        mu = np.zeros(3)

        samples, n_samples_dir = self.sample_poses_deterministic(
            pose=scan_match_pose,
            sigma_xy=sigma_xy,
            sigma_theta=sigma_theta,
            n_samples_dir=n_samples_dir,
        )

        # Predict particle pose based on odometry and old particle pose 
        dl, dr = odom
        pred_pose = motion_model.predict_pose(
            pose=particle.pose,
            dl=dl,
            dr=dr,
        )

        # Compute motion probabilities
        motion_probs = motion_model.motion_probability_batch(
            x_new=samples,
            x_prev=pred_pose,
        )
        
        # Compute measurement probabilities  
        log_likelihoods = np.empty_like(motion_probs)
        for i, sample in enumerate(samples):
            score, log_likelihood, matched_count  = measurement_model.gmapping_likelihood(
                pose=sample,
                measurements=measurements,
                ogm=particle.scan_matcher.ogm,
                usable_range=particle.scan_matcher.max_sensor_range,
                kernel_size=meas_kernel_size,
                fullness_threshold=particle.scan_matcher.occ_thres,
                free_threshold=particle.scan_matcher.occ_thres,
                gaussian_sigma=gaussian_sigma,
                free_cell_ratio=np.sqrt(2.0),
            )

            log_likelihoods[i] = log_likelihood

        meas_probs = np.exp(log_likelihoods - np.max(log_likelihoods))

        # Transfer probs into log space
        # log_meas_probs = np.log(meas_probs + 1e-12)
        # log_motion_probs = np.log(motion_probs + 1e-12)

        # # Scale and combine log probs into log weights
        # log_weights = alpha * log_motion_probs + beta * log_meas_probs
        # log_weights = log_weights - np.max(log_weights)
        # weights = np.exp(log_weights)

        # COmpute xj weights
        weights = meas_probs * motion_probs
                       
        # Vectorized computation of mu and cov
        norm = np.sum(weights)

        if (not np.isfinite(norm)) or norm <= 1e-12:
            # Fallback when all sample weights collapse to zero/invalid values.
            mu = np.asarray(scan_match_pose, dtype=float)
            cov = 1e-6 * np.eye(3)
            weights = np.ones(samples.shape[0], dtype=float)
            meas_probs = np.ones(samples.shape[0], dtype=float)
            motion_probs = np.ones(samples.shape[0], dtype=float)
            return mu, cov, 1e-12, samples, weights, meas_probs, motion_probs, pred_pose

        # Compute mu
        mu = np.sum(samples * weights[:, None], axis=0) / norm

        # Compute covariance matrix
        # Compute deviation from the mean
        x_minus_mu = samples - mu
        # Ensure valid angles
        x_minus_mu[:, self.IDX_THETA] = np.arctan2(np.sin(x_minus_mu[:, self.IDX_THETA]), np.cos(x_minus_mu[:, self.IDX_THETA]))
        # Compute noralized covariance
        cov = (weights[:, None] * x_minus_mu).T @ x_minus_mu / norm

        # Ensure covariance matrix is positive definite by adding small values to diagonal
        cov += 1e-6 * np.eye(3)

        return mu, cov, norm, samples, weights, meas_probs, motion_probs, pred_pose
    
    
    def compute_proposal_param_batch(
        self,
        scan_match_pose: Pose2D,
        particle: Particle,
        odom: Tuple[float, float],
        measurements: List[Tuple[float, float]],
        neighbor: NearestNeighbors,
        motion_model: MotionModel,
        measurement_model: MeasurementModel,
        sigma_xy: float=1.0,
        sigma_theta: float=1.0,
        n_samples: int=3,
        alpha: float=0.5,
        beta: float=2.0,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Pose2D]:
        '''
        Proposal computation with deterministic sampling around scan match pose. Motion and measurement probabilities are 
        computed in batch to speedup process. The gmapping measurement model is used here!
        '''
        # Define vars
        norm = 0.0
        mu = np.zeros(3)

        samples, n_samples = self.sample_poses_deterministic(
            pose=scan_match_pose,
            sigma_xy=sigma_xy,
            sigma_theta=sigma_theta,
            n_samples_dir=n_samples,
        )

        # Predict particle pose based on odometry and old particle pose 
        dl, dr = odom
        pred_pose = motion_model.predict_pose(
            pose=particle.pose,
            dl=dl,
            dr=dr,
        )

        # Compute probability and add to normalizer 
        meas_probs = measurement_model.likelihood_batch(
            poses=samples,
            measurements=measurements,
            scan_matcher= particle.scan_matcher,
            neighbor=neighbor,
        )


        motion_probs = motion_model.motion_probability_batch(
            x_new=samples,
            x_prev=pred_pose,
        )

        # Transfer probs into log space
        # log_meas_probs = np.log(meas_probs + 1e-12)
        # log_motion_probs = np.log(motion_probs + 1e-12)

        # # Scale and combine log probs into log weights
        # log_weights = alpha * log_motion_probs + beta * log_meas_probs
        # log_weights = log_weights - np.max(log_weights)
        # weights = np.exp(log_weights)

        weights = meas_probs * motion_probs
                       
        # Vectorized computation of mu and cov
        norm = np.sum(weights)

        if (not np.isfinite(norm)) or norm <= 1e-12:
            # Fallback when all sample weights collapse to zero/invalid values.
            mu = np.asarray(scan_match_pose, dtype=float)
            cov = 1e-6 * np.eye(3)
            weights = np.ones(samples.shape[0], dtype=float)
            meas_probs = np.ones(samples.shape[0], dtype=float)
            motion_probs = np.ones(samples.shape[0], dtype=float)
            return mu, cov, 1e-12, samples, weights, meas_probs, motion_probs, pred_pose

        # Compute mu
        mu = np.sum(samples * weights[:, None], axis=0) / norm

        # Compute covariance matrix
        # Compute deviation from the mean
        x_minus_mu = samples - mu
        # Ensure valid angles
        x_minus_mu[:, self.IDX_THETA] = np.arctan2(np.sin(x_minus_mu[:, self.IDX_THETA]), np.cos(x_minus_mu[:, self.IDX_THETA]))
        # Compute noralized covariance
        cov = (weights[:, None] * x_minus_mu).T @ x_minus_mu / norm

        # Ensure covariance matrix is positive definite by adding small values to diagonal
        cov += 1e-6 * np.eye(3)

        return mu, cov, norm, samples, weights, meas_probs, motion_probs, pred_pose
    

    def compute_proposal_param_batch_copy(
        self,
        scan_match_pose: Pose2D,
        particle: Particle,
        odom: Tuple[float, float],
        measurements: List[Tuple[float, float]],
        neighbor: NearestNeighbors,
        motion_model: MotionModel,
        measurement_model: MeasurementModel,
        sigma_xy: float=1.0,
        sigma_theta: float=1.0,
        n_samples: int=3,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Pose2D]:
        '''
        Proposal computation with deterministic sampling around scan match pose. Motion and measurement probabilities are 
        computed in batch to speedup process. The old NN based measurment model is used here with clipped distances!
        '''
        # Define vars
        norm = 0.0
        mu = np.zeros(3)

        samples, n_samples = self.sample_poses_deterministic(
            pose=scan_match_pose,
            sigma_xy=sigma_xy,
            sigma_theta=sigma_theta,
            n_samples_dir=n_samples,
        )

        # Predict particle pose based on odometry and old particle pose 
        dl, dr = odom
        pred_pose = motion_model.predict_pose(
            pose=particle.pose,
            dl=dl,
            dr=dr,
        )

        # Compute probability and add to normalizer 
        meas_probs = measurement_model.likelihood_batch_copy(
            poses=samples,
            measurements=measurements,
            scan_matcher= particle.scan_matcher,
            neighbor=neighbor,
        )

        motion_probs = motion_model.motion_probability_batch(
            x_new=samples,
            x_prev=pred_pose,
        )

        weights = meas_probs * motion_probs
                       
        # Vectorized computation of mu and cov
        norm = np.sum(weights)

        if (not np.isfinite(norm)) or norm <= 1e-12:
            # Fallback when all sample weights collapse to zero/invalid values.
            mu = np.asarray(scan_match_pose, dtype=float)
            cov = 1e-6 * np.eye(3)
            weights = np.ones(samples.shape[0], dtype=float)
            meas_probs = np.ones(samples.shape[0], dtype=float)
            motion_probs = np.ones(samples.shape[0], dtype=float)
            return mu, cov, 1e-12, samples, weights, meas_probs, motion_probs, pred_pose

        # Compute mu
        mu = np.sum(samples * weights[:, None], axis=0) / norm

        # Compute covariance matrix
        # Compute deviation from the mean
        x_minus_mu = samples - mu
        # Ensure valid angles
        x_minus_mu[:, self.IDX_THETA] = np.arctan2(np.sin(x_minus_mu[:, self.IDX_THETA]), np.cos(x_minus_mu[:, self.IDX_THETA]))
        # Compute noralized covariance
        cov = (weights[:, None] * x_minus_mu).T @ x_minus_mu / norm

        # Ensure covariance matrix is positive definite by adding small values to diagonal
        cov += 1e-6 * np.eye(3)

        return mu, cov, norm, samples, weights, meas_probs, motion_probs, pred_pose    


    def sample_from_proposal(self, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        '''
        Sample a new pose from a Gaussian distribution with mean mu and covariance sigma. Ensures angles are normalized.
        '''
        new_pose = np.random.multivariate_normal(mean=mu, cov=cov)
        new_pose[self.IDX_THETA] = np.arctan2(np.sin(new_pose[self.IDX_THETA]), np.cos(new_pose[self.IDX_THETA]))
        return new_pose
    

    def estimate_proposal(
        self,
        scan_match_pose: Pose2D,
        particle: Particle,
        odom: Tuple[float, float],
        measurements: List[Tuple[float, float]],
        neighbor: NearestNeighbors,
        motion_model: MotionModel,
        measurement_model: MeasurementModel,
        sigma_xy: float=1.0,
        sigma_theta: float=1.0,
        n_samples: int=3,
        meas_kernel_size: int=1,
        gaussian_sigma: float=0.05,
        alpha: float=0.5,
        beta: float=2.0,
    ) -> Tuple[np.ndarray, float, dict]:
        '''

        '''

        # Compute proposal parameters old variant (NN tree distances + clip)
        mu, cov, p_weight, xjs, xj_weights, meas_probs, motion_probs, pred_pose = self.compute_proposal_param_batch_copy(
            scan_match_pose=scan_match_pose,
            particle=particle,
            odom=odom,
            measurements=measurements,
            neighbor=neighbor,
            motion_model=motion_model,
            measurement_model=measurement_model,
            sigma_xy=sigma_xy,
            sigma_theta=sigma_theta,
            n_samples=n_samples,
        )

        # Compute proposal params
        # mu, cov, p_weight, xjs, xj_weights, meas_probs, motion_probs, pred_pose = self.compute_proposal_param_batch(
        #     scan_match_pose=scan_match_pose,
        #     particle=particle,
        #     odom=odom,
        #     measurements=measurements,
        #     neighbor=neighbor,
        #     motion_model=motion_model,
        #     measurement_model=measurement_model,
        #     sigma_xy=sigma_xy,
        #     sigma_theta=sigma_theta,
        #     n_samples=n_samples,
        #     alpha=alpha,
        #     beta=beta,
        # )

        # mu, cov, p_weight, xjs, xj_weights, meas_probs, motion_probs, pred_pose = self.compute_proposal_param_gmapping(
        #     scan_match_pose=scan_match_pose,
        #     particle=particle,
        #     odom=odom,
        #     measurements=measurements,
        #     motion_model=motion_model,
        #     measurement_model=measurement_model,
        #     meas_kernel_size=meas_kernel_size,
        #     gaussian_sigma=gaussian_sigma,
        #     sigma_xy=sigma_xy,
        #     sigma_theta=sigma_theta,
        #     n_samples_dir=n_samples,
        #     alpha=alpha,
        #     beta=beta,
        # )

        # Store raw proposal diagnostics for downstream evaluation.
        info = {
            "prop_mu": mu,
            "prop_cov_matrix": cov,
            "scan_match_pose": np.asarray(scan_match_pose, dtype=float),
            "pred_pose": np.asarray(pred_pose, dtype=float),
            "xjs": xjs,
            "xj_weights": xj_weights,
            "motion_probs": motion_probs,
            "meas_probs": meas_probs,
        }

        # Sample pose
        # new_pose = self.sample_from_proposal(mu, cov)
        
        # TODO: Keep this for the moment until we found better solution. Chat GPT recommends this version due to better results
        new_pose = mu 
        
        return new_pose, p_weight, info


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
        

    def compute_proposal_params_range_finder_model(
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
        Proposal computation with deterministic sampling around scan match pose. Motion and measurement probabilities are 
        computed in batch to speedup process. The old NN based measurment model is used here with clipped distances!
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
        log_likelihoods = np.empty(shape=(samples.shape[0],))
        for i, sample in enumerate(samples):
            results = measurement_model.likelihood(
                pose=sample,
                measurements=measurements,
                ogm=particle.scan_matcher.ogm,
            )
            # Extract log-likelihood 
            log_likelihood = results.get("log_likelihood", INVALID_LOG_LIKELIHOOD)
            log_likelihoods[i] = log_likelihood

            # Update measurement model counters for diagnostics
            self.update_measurement_model_counters(results)
            
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


    def estimate_proposal_range_finder(
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
        # Rest proposal timings
        self.t_sample_poses = 0.0
        self.t_pred_poses = 0.0
        self.t_motion_model = 0.0
        self.t_meas_model = 0.0
        self.t_compute_prop_params = 0.0
        self.t_sample_from_prop = 0.0

        # Compute proposal parameter with ray tracing 
        mu, cov, norm, xjs, xj_weights, meas_probs, motion_probs, pred_pose, log_eta = self.compute_proposal_params_range_finder_model(
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
        new_p_pose = mu

        # new_p_pose = self.sample_from_proposal(mu, cov)
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