#!/usr/bin/env python3
from typing import Tuple
import numpy as np


from slam.infrastructure.defs import Pose2D
from slam.scan_matcher.scan_matcher import ScanMatcher


class Particle:
    '''
    Class representing a single particle.  
    '''
    def __init__(self, pose: Pose2D, weight: float, scan_matcher: ScanMatcher):
        self.pose = pose
        self.weight = weight
        self.scan_matcher = scan_matcher


    def copy(self):
        '''
        Creates a copy of the particle. The scan matcher is also copied to ensure that the particles are
        independent of each other.
        '''
        return Particle(
            pose=self.pose,
            weight=self.weight,
            scan_matcher=self.scan_matcher.copy()
        )