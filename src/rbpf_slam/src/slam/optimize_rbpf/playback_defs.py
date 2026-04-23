from typing import List, Tuple, Any
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StepData:
    '''
    Data storage to perform one step in scan matching 
    '''
    t: float
    dl: float
    dr: float
    scan: List[Tuple[float, float]]   # (range, bearing)
    true_pose: Tuple[float, float, float]  # (x, y, yaw)


