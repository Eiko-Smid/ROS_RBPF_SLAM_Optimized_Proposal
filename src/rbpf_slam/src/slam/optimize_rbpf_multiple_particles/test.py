from dataclasses import fields
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Type,
    Union,
    Tuple,
)

from ..infrastructure.defs import Pose2D

import numpy as np
import pandas as pd

from abc import abstractmethod, ABC


class MeasurementModel(ABC):
    @abstractmethod
    def likelihood(
        self, 
        pose: Pose2D, 
        measurements: List[Tuple[float, float]],
        **kwargs,
    ) -> Union[float, Dict]:
        '''
        Compute the likelihood of a pose given a set of measurements.
        
        '''
        
        raise NotImplementedError


class BeamRangeFinderModel(MeasurementModel):
    def likelihood(
        self,
        pose: Pose2D,
        measurements: List[Tuple[float, float]],
        param_1: Any
    ):
        return 0.0


class LikelihoodModel(MeasurementModel):
    def likelihood(
        self,
        pose: Pose2D,
        measurements: List[Tuple[float, float]], 
    ):
        
        return 1.0


class Proposal:
    def compute_proposal(
        self,
        pose: Pose2D,
        measurements: List[Tuple[float, float]], 
        meas_model: MeasurementModel,
        # param_1: Any
    ):
        param_1 = 2.0

        likelihood = meas_model.likelihood(
            pose=pose,
            measurements=measurements,
        )
        return likelihood



def main():
    pose = (1.0, 1.0, 3.14)
    measurements = [(1.0, 2.0), (3.0, 4.0)]
    # meas_model = BeamRangeFinderModel()
    meas_model = LikelihoodModel()

    proposal = Proposal()
    likelihood = proposal.compute_proposal(
        pose=pose,
        measurements=measurements,
        meas_model=meas_model,
    )

    print(f"Likelihood: {likelihood}")


if __name__ == "__main__":
    main()