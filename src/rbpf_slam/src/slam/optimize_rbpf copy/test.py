#!/usr/bin/env python3
from dataclasses import dataclass

from math import atan2, cos, sin, floor
from typing import Tuple
import numpy as np

import matplotlib.pyplot as plt
import time

from ..rbpf.proposal import ProposalEstimator


@dataclass
class Pose2D:
    x: float
    y: float
    theta: float


def visualize_proposal_samples(samples: np.ndarray, true_pose: Tuple[float, float, float]):
    x, y, theta = true_pose

    plt.figure()
    plt.scatter(samples[:, 0], samples[:, 1], label="Proposal Samples", alpha=0.5)
    plt.scatter(x, y, color="red", label="True Pose", marker="x")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Proposal Samples and True Pose")
    plt.legend()
    plt.axis("equal")
    plt.grid()
    plt.show()


def generate_proposal_samples():
    true_pose = (1.0, 2.0, np.radians(90))
    sigma_xy = 0.1
    sigma_theta = 0.02
    n_samples_dir = 3

    proposal_estimator = ProposalEstimator()
    samples, n_xjs = proposal_estimator.sample_poses_deterministic(
        pose=true_pose,
        sigma_xy=sigma_xy,
        sigma_theta=sigma_theta,
        n_samples_dir=n_samples_dir,
    )

    return samples, true_pose



def rank_model_probs():
    weights = np.array([0.5, 0.4, 0.3])
    # xj_true_pose_err = np.array([0.1, 0.2, 0.3])
    xj_true_pose_err = np.array([0.3, 0.2, 0.1])
    
    # get idx fof max weights
    max_weight_idx = np.argmax(weights)

    # Pseudo sort pose err from low to high
    order = np.argsort(xj_true_pose_err)

    # Compute rank
    rank = int(np.where(order == max_weight_idx)[0][0]) + 1

    # Compute score
    N = len(xj_true_pose_err)
    if N == 1:
        rank_score = 1.0
    else:
        rank_score = 1.0 - (rank - 1) / (N - 1)
    
    return rank_score


def test():
    MEASUREMENT_STDDEV = 0.03
    max_sensor_range=10.0 + MEASUREMENT_STDDEV 

    print(max_sensor_range)

    MEASUREMENT_STDDEV = None
    max_sensor_range=10.0 + MEASUREMENT_STDDEV 
    print(max_sensor_range)



def main():
    # rank_model_probs()
    test()


if __name__=="__main__":
    main()