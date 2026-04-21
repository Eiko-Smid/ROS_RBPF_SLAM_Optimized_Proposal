from typing import List, Tuple
import numpy as np

from slam.infrastructure.defs import Pose2D


class ProposalEstimator:
    @staticmethod
    def sample_poses(pose: Pose2D, sigma_xy: float, n_samples: int) -> np.ndarray:
        x, y, theta = pose

        samples = np.zeros((n_samples, 3))
        samples[:, 0] = np.random.normal(x, sigma_xy, n_samples)
        samples[:, 1] = np.random.normal(y, sigma_xy, n_samples)
        angles = np.random.normal(theta, sigma_xy, n_samples)
        samples[:, 2] = np.arctan2(np.sin(angles), np.cos(angles))

        return samples


    @staticmethod
    def compute_proposal(particles, measurement):
        pass