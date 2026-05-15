
from typing import List, Tuple

import numpy as np

from .measurement_model import MeasurementModel
from sklearn.neighbors import NearestNeighbors

from slam.scan_matcher.scan_matcher import ScanMatcher
from slam.infrastructure.defs import Pose2D


class LikelihoodFiledModel(MeasurementModel):
    def __init__(self, sigma: float=0.1) -> None:
        self.sigma = sigma
         
    
    def likelihood(
        self,
        pose: Pose2D,
        measurements: List[Tuple[float, float]],
        scan_matcher: ScanMatcher,
        neighbor: NearestNeighbors,
    ) -> float:
        
        # Safety checks
        if scan_matcher is None or neighbor is None:
            return 1e-9

        if len(measurements) < 3:
            return 1e-9

        # Transform to points
        scan_points = scan_matcher.transform_measurements_to_points(
            pose=pose,
            measurements=measurements,
        )

        # Check if enough scan points available
        if len(scan_points) < 3:
            return 1e-9

        # Get distances to nearest neighbor for every scan point
        distances, _ = neighbor.kneighbors(scan_points, n_neighbors=1)
        distances = distances[:, 0]

        # Clip distances to weight bad correspondences lower
        # TODO: Define paramter for clipping. Also test without clipping
        distances = np.clip(distances, 0.0, 1.0)

        # Use mean error to increase measrument likelihood robustness to outliers
        mean_error = np.mean((distances / self.sigma) ** 2)
        prob = np.exp(-0.5 * mean_error)  

        # Likelihood (Gaussian)
        # prob = np.exp(
        #     -0.5 * np.sum((distances / self.sigma) ** 2)
        # )

        return float(prob)

    
    def likelihood_batch(
        self,
        poses: np.ndarray,
        measurements: List[Tuple[float, float]],
        scan_matcher: ScanMatcher,
        neighbor: NearestNeighbors,
    ) -> np.ndarray:

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

        # TODO: Add clipping again later on
        distances = np.clip(distances, 0.0, 1.0)

        mean_error = np.mean(
            (distances / self.sigma) ** 2,
            axis=1,
        )

        k = 5.0
        scaled_mean = -0.5 * k * mean_error
        
        # probs = np.exp(-0.5 * mean_error)
        probs = np.exp(scaled_mean)

        return probs