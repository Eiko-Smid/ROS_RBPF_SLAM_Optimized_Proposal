from typing import List, Tuple, Any

import numpy as np
from sklearn.neighbors import NearestNeighbors


from slam.scan_matcher.scan_matcher import ScanMatcher
from slam.infrastructure.defs import Pose2D


class MeasurementModel:
    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma


    def likelihood(
            self,
            scan_matcher: ScanMatcher,
            pose: Pose2D,
            measurements: List[Tuple[float, float]],
            neighbor: NearestNeighbors,
            every_nth_measurement: int = 5,
    ):
        # Check if we have enough measurements
        if len(measurements) < 3:
            return 1e-9
        
        # Use very n'th measurement to speed up the likelihood computation. 
        measurements = measurements[::every_nth_measurement]
        
        # Transform scan (range, bearing) -> points
        scan_points = scan_matcher.transform_measurements_to_points(
            pose=pose,
            measurements=measurements,
        )
        if len(scan_points) < 3:
            return 1e-9

        # Find NN
        distances, _ = neighbor.kneighbors(scan_points, n_neighbors=1)
        distances = distances[:, 0]

        # Compute measurement likelihood
        # TODO: Replace with log likelihood to avoid numerical issues
        prob = np.exp(
            -0.5 * np.sum((distances/self.sigma)**2)
        )

        return prob
