
from typing import List, Tuple

import numpy as np

from sklearn.neighbors import NearestNeighbors

from slam.scan_matcher.scan_matcher import ScanMatcher
from .measurement_model import MeasurementModel
from ..infrastructure.defs import Pose2D

'''
Nearest Neighbor based implementation of the likelihood field model. 

TODO: 
        If used later on we need to make the compuation of the max_distances dependend on grid resolution. 
        max_distance = np.sqrt(2.0) * (kernel_size + 0.5) * resolution
'''

class LikelihoodFiledModelNN(MeasurementModel):
    def __init__(self, sigma: float=0.1) -> None:
        self.sigma = sigma


    def likelihood(
        self,
        pose: Pose2D,
        measurements: List[Tuple[float, float]],
        scan_matcher: ScanMatcher,
        neighbor: NearestNeighbors,
    ) -> float:
        '''
        Compute the measurement likelihood for the given pose.

        Parameters
        ----------
        pose : Pose2D
            The pose for which to compute the likelihood.
        measurements : List[Tuple[float, float]]
            The list of measurements (e.g., LiDAR scan points) to compare against the scan matcher.
        scan_matcher : ScanMatcher
            The scan matcher used to compute the likelihood.
        neighbor : NearestNeighbors
            The nearest neighbor model used to find the closest points in the map.

        Returns
        -------
        float
            The likelihood value for the given pose.
        '''
        # Keep the fallback behavior consistent with likelihood_batch.
        if scan_matcher is None or neighbor is None:
            return 1e-9

        if len(measurements) < 3:
            return 1e-9

        null_likelihood = -0.5
        no_hit = null_likelihood / self.sigma
        max_distance = 0.3

        # Convert once so ranges and bearings can be accessed without two
        # separate Python-level comprehensions.
        measurement_array = np.asarray(measurements, dtype=np.float64)
        ranges = measurement_array[:, 0]
        bearings = measurement_array[:, 1]

        local_x = ranges * np.cos(bearings)
        local_y = ranges * np.sin(bearings)

        px, py, pt = pose
        c = np.cos(pt)
        s = np.sin(pt)

        # Preallocate the query array and transform all scan endpoints in one
        # vectorized operation. Only one nearest-neighbor query is required.
        world_points = np.empty((len(measurements), 2), dtype=np.float64)
        world_points[:, 0] = px + c * local_x - s * local_y
        world_points[:, 1] = py + s * local_x + c * local_y

        distances = neighbor.kneighbors(
            world_points,
            n_neighbors=1,
            return_distance=True,
        )[0][:, 0]

        valid_mask = distances <= max_distance
        valid_distances = distances[valid_mask]

        log_likelihood = -(valid_distances @ valid_distances) / self.sigma
        log_likelihood += no_hit * np.count_nonzero(~valid_mask)

        # A long scan can produce a log-likelihood below the representable
        # floating-point range. Returning the smallest positive float avoids
        # turning a valid, very unlikely pose into an exact zero weight.
        min_log = np.log(np.finfo(np.float64).tiny)
        return float(np.exp(max(log_likelihood, min_log)))


    def likelihood_batch(
        self,
        poses: np.ndarray,
        measurements: List[Tuple[float, float]],
        scan_matcher: ScanMatcher,
        neighbor: NearestNeighbors,
    ) -> np.ndarray:
        '''
        Computes the measurement likelihoods for the given poses and measurements.

        Parameters
        ----------
        poses : np.ndarray
            An array of poses for which to compute the likelihoods. Shape should be (N, 3) where N is the number of poses.
        measurements : List[Tuple[float, float]]
            The list of measurements (e.g., LiDAR scan points) to compare against the scan matcher.
        scan_matcher : ScanMatcher
            The scan matcher used to compute the likelihoods.
        neighbor : NearestNeighbors
            The nearest neighbor model used to find the closest points in the map.
        
        Returns
        -------
        np.ndarray
            The likelihoods corresponding to each pose.
        '''
        # Compute no git lieklihood for punishment
        null_likelihood = -0.5
        no_hit = null_likelihood / self.sigma
        # Define max distance 
        max_distance = 0.3

        n_poses = poses.shape[0]

        if scan_matcher is None or neighbor is None:
            return np.full(n_poses, 1e-9)

        if len(measurements) < 3:
            return np.full(n_poses, 1e-9)

        # --------------------------------------------------
        # Precompute local scan points once
        # --------------------------------------------------

        ranges = np.array([m[0] for m in measurements])
        bearings = np.array([m[1] for m in measurements])

        local_x = ranges * np.cos(bearings)
        local_y = ranges * np.sin(bearings)

        n_beams = len(ranges)

        # --------------------------------------------------
        # Transform all poses at once
        # --------------------------------------------------

        px = poses[:, 0][:, None]
        py = poses[:, 1][:, None]
        pt = poses[:, 2][:, None]

        c = np.cos(pt)
        s = np.sin(pt)

        world_x = px + c * local_x - s * local_y
        world_y = py + s * local_x + c * local_y

        # shape -> (Nposes * Nbeams, 2)
        all_points = np.stack(
            [world_x.reshape(-1), world_y.reshape(-1)],
            axis=1,
        )

        # --------------------------------------------------
        # ONE nearest-neighbor call
        # --------------------------------------------------

        distances, _ = neighbor.kneighbors(
            all_points,
            n_neighbors=1,
        )

        distances = distances[:, 0]

        # reshape back
        distances = distances.reshape(n_poses, n_beams)

        # Create mask to access all valid distances
        valid_mask = distances <= max_distance

        # Compute likelihood of valid distances
        valid_dist2 = np.where(valid_mask, distances ** 2, 0.0)
        log_likelihoods = np.sum(-valid_dist2 / self.sigma, axis=1)

        # Add likelihood for invalid distances
        n_invalid = np.sum(~valid_mask, axis=1)
        log_likelihoods += no_hit * n_invalid

        # Convert log-likelihoods to stable positive weights/probs
        log_likelihoods = log_likelihoods - np.max(log_likelihoods)
        probs = np.exp(log_likelihoods)

        return probs
