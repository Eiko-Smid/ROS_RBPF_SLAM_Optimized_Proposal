from typing import List, Tuple, Any

from abc import ABC, abstractmethod

import numpy as np
from sklearn.neighbors import NearestNeighbors


from slam.scan_matcher.scan_matcher import ScanMatcher
from slam.infrastructure.defs import Pose2D


class MeasurementModel(ABC):
    '''
    Base class for 2D measurement models. Defines the interface for computing the likelihood of a pose given 
    a set of measurements and a scan matcher.   
    The likelihood function should return a probability value indicating how well the given pose explains
    the measurements.
    '''
    @abstractmethod
    def likelihood(
            self, 
            pose: Pose2D,
            measurements: List[Tuple[float, float]],
            **kwargs,
    ) -> float:
        '''
        Compute the likelihood of a pose given a set of measurements and a scan matcher.

        Parameters:
        pose: Pose2D  
            The pose for which to compute the likelihood.
        measurements: List[Tuple[float, float]]
            The list of measurements (e.g., LiDAR scan points) to compare against the scan matcher.
        
        '''
        raise NotImplementedError
    
