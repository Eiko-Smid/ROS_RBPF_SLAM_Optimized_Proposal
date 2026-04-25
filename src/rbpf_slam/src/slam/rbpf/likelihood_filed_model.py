
from typing import List, Tuple, Any

import numpy as np

from .measurement_model import MeasurementModel
from sklearn.neighbors import NearestNeighbors

from slam.scan_matcher.scan_matcher import ScanMatcher
from slam.infrastructure.defs import Pose2D


class LikelihoodFiledModel(MeasurementModel):
    def __init__(self, sigma: float=0.1, every_nth_measurement: int = 5) -> None:
        self.sigma = sigma
        self.every_nth_measurement: int = every_nth_measurement
         
    
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

        # Subsample measurements for speed
        measurements = measurements[::self.every_nth_measurement]

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
        distances = np.clip(distances, 0.0, 1.0)

        # Use mean error to increase measrument likelihood robustness to outliers
        mean_error = np.mean((distances / self.sigma) ** 2)
        prob = np.exp(-0.5 * mean_error)  

        # Likelihood (Gaussian)
        # prob = np.exp(
        #     -0.5 * np.sum((distances / self.sigma) ** 2)
        # )

        return float(prob)

    