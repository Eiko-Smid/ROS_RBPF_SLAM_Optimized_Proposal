from typing import List, Tuple
import numpy as np

from sklearn.neighbors import NearestNeighbors

from slam.infrastructure.defs import Pose2D
from slam.rbpf.particle import Particle
from slam.rbpf.motion_model import MotionModel
from slam.rbpf.measurement_model import MeasurementModel


class ProposalEstimator:
    IDX_x=0
    IDX_y=1
    IDX_THETA=2


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
       
        # Compute measurement model likelihoods for each sample  
        log_likelihoods = np.empty(shape=(samples.shape[0],))
        for i, sample in enumerate(samples):
            results = measurement_model.likelihood(
                pose=sample,
                measurements=measurements,
                ogm=particle.scan_matcher.ogm,
            )
            log_likelihood = results.get("log_likelihood", -np.inf)

            log_likelihoods[i] = log_likelihood
            
        # Convert log-likelihoods to probabilities in save manner
        max_log_likelihood = np.max(log_likelihoods)

        if not np.isfinite(max_log_likelihood):
            mu = np.asarray(scan_match_pose, dtype=float)
            cov = 1e-6 * np.eye(3)
            weights = np.ones(samples.shape[0], dtype=float)
            meas_probs = np.ones(samples.shape[0], dtype=float)
            motion_probs = np.ones(samples.shape[0], dtype=float)
            norm =  1e-12
            log_eta = -np.inf
            return mu, cov, norm, samples, weights, meas_probs, motion_probs, pred_pose, log_eta
        else:
            meas_probs = np.exp(log_likelihoods - max_log_likelihood)

        # Compute motion probs
        motion_probs = motion_model.motion_probability_batch(
            x_new=samples,
            x_prev=pred_pose,
        )

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
            log_eta = -np.inf
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

        return mu, cov, norm, samples, weights, meas_probs, motion_probs, pred_pose, log_eta 



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

        # Estimate new particle pose
        # TODO: Repalce that at the end to get different poses for teh particles. THink about how to do/sample 
        new_p_pose = mu

        return new_p_pose, log_eta, info