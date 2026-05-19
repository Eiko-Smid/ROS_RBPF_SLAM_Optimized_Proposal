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
    

    @abstractmethod
    def likelihood_batch(
        self, 
        poses: np.ndarray,
        measurements: List[Tuple[float, float]],
        **kwargs,
    ) -> np.ndarray:
        '''
        Compute the likelihood of multiple poses given a set of measurements and a scan matcher.

        Parameters:
        poses: np.ndarray
            An array of poses for which to compute the likelihood. Shape should be (N, 3) where N is the number of poses.
        measurements: List[Tuple[float, float]]
            The list of measurements (e.g., LiDAR scan points) to compare against the scan matcher.
        
        Returns:
        np.ndarray
            An array of likelihood values corresponding to each pose. Shape should be (N,).
        '''
        raise NotImplementedError
    
    
    @abstractmethod
    def likelihood_batch_copy(
        self, 
        poses: np.ndarray,
        measurements: List[Tuple[float, float]],
        **kwargs,
    ) -> np.ndarray:
        '''
        Compute the likelihood of multiple poses given a set of measurements and a scan matcher.

        Parameters:
        poses: np.ndarray
            An array of poses for which to compute the likelihood. Shape should be (N, 3) where N is the number of poses.
        measurements: List[Tuple[float, float]]
            The list of measurements (e.g., LiDAR scan points) to compare against the scan matcher.
        
        Returns:
        np.ndarray
            An array of likelihood values corresponding to each pose. Shape should be (N,).
        '''
        raise NotImplementedError
    

    @abstractmethod
    def gmapping_likelihood(
        self, pose: Pose2D,
        measurements: List[Tuple[float, float]],
        **kwargs,
    ) -> Tuple[float, float, int]:
        '''
        gmapping likelihood function that computes the likelihood of a pose given a set of measurements and an OGM map.

            - 
        '''
        raise NotImplementedError
