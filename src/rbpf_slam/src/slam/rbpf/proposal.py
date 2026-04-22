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


    def sample_poses(self, pose: Pose2D, sigma_xy: float, n_samples: int) -> np.ndarray:
        x, y, theta = pose

        samples = np.zeros((n_samples, 3))
        samples[:, self.IDX_x] = np.random.normal(x, sigma_xy, n_samples)
        samples[:, self.IDX_y] = np.random.normal(y, sigma_xy, n_samples)
        angles = np.random.normal(theta, sigma_xy, n_samples)
        samples[:, self.IDX_THETA] = np.arctan2(np.sin(angles), np.cos(angles))

        return samples


    def compute_proposal_param(
            self,
            scan_match_pose: Pose2D,
            particle: Particle,
            measurements: List[Tuple[float, float]],
            neighbor: NearestNeighbors,
            motion_model: MotionModel,
            measurement_model: MeasurementModel,
            sigma_xy: float=1.0,
            n_samples: int=10,
    ):
        # Define vars
        norm = 0.0
        mu = np.zeros(3)
        weights = np.zeros(shape=(n_samples))

        # Sample k new poses around scan matcher pose
        samples = self.sample_poses(
            pose=scan_match_pose,
            sigma_xy=sigma_xy,
            n_samples=n_samples,
        )        

        # Compute Gaussian parameters µ and Cov
        for i in range(samples.shape[0]):
            xj = samples[i, :]
            meas_prob = measurement_model.likelihood(
                scan_matcher=particle.scan_matcher,
                pose=xj,
                measurements=measurements,
                neighbor=neighbor,
                every_nth_measurement=5,
            )

            motion_prob = motion_model.motion_probability(
                x_new=xj,
                x_prev=particle.pose,
            )

            # COmpute probability and add to normalizer 
            w = meas_prob * motion_prob
            weights[i] = w            

        # Vectorized computation of mu and cov
        norm = np.sum(weights)

        # Compute mu
        mu = np.sum(samples * weights[:, None]) / norm

        # Compute covariance matrix
        # Compute deviation from the mean
        x_minus_mu = samples - mu
        # Ensure valid angles
        x_minus_mu[:, self.IDX_THETA] = np.arctan2(np.sin(x_minus_mu[:, self.IDX_THETA]), np.cos(x_minus_mu[:, self.IDX_THETA]))
        # Compute noralized covariance
        cov = (weights[:, None] * x_minus_mu).T @ x_minus_mu / norm

        # Ensure covariance matrix is positive definite by adding small values to diagonal
        cov += 1e-6 * np.eye(3)

        return mu, cov
    

    def sample_pose(self, mu, cov):
        '''
        Sample a new pose from a Gaussian distribution with mean mu and covariance sigma. Ensures angles are normalized.
        '''
        new_pose = np.random.multivariate_normal(mean=mu, cov=cov)
        new_pose[self.IDX_THETA] = np.arctan2(np.sin(new_pose[self.IDX_THETA]), np.cos(new_pose[self.IDX_THETA]))
        return new_pose