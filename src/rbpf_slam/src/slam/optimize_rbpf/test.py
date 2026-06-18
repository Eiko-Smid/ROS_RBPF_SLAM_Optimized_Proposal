#!/usr/bin/env python3

# import debugpy
# debugpy.listen(("localhost", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

from dataclasses import dataclass

from math import atan2, cos, sin, floor
from typing import Tuple
import numpy as np

import matplotlib.pyplot as plt
import time

from ..rbpf.proposal import ProposalEstimator
from ..rbpf.rbpf import RBPF


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
    log_p_weight = -np.inf

    log_weights = [1.2, 0.3, 5.0]
    log_weights.append(log_p_weight)

    log_weights_np = np.asarray(log_weights)
    old_log_weights_np = np.array([0.2, 0.1, 0.2, 0.5])

    #  normalize weights
    norm_weights = RBPF.normalize_weights(
        old_weights=old_log_weights_np,
        log_weight_increments=log_weights_np
    )

    print(log_weights_np)

    print(f"\nNormalized weights:\n{norm_weights}")





class RangeFinderModel:
    def __init__(
            self,
            w_hit: float,
            w_short: float,
            w_max: float,
            w_rand: float,
            lambda_short: float = 0.1
    ):
        self.w_hit = w_hit
        self.w_short = w_short
        self.lambda_short = lambda_short
        self.w_max = w_max
        self.w_rand = w_rand


    def correct_range_likelihood(self, r: float, r_pred: float) -> float:
        norm = 1 / (self.w_hit * np.sqrt(2 * np.pi))
        likelihood = norm * np.exp(-0.5 * ((r - r_pred) / self.w_hit) ** 2)
        return likelihood
    
    
    def unexpected_likelihood(self, r: float, r_pred: float) -> float:
        norm = 1 / (1 - np.exp(-self.lambda_short * r_pred))
        likelihood = norm * self.lambda_short * np.exp(-self.lambda_short * r)
        return likelihood
    

    def failure_likelihood(self, r: float) -> float:
        if r == self.z_max:
            return 1.0
        else:
            return 0.0
        

    def random_likelihood(self, r: float, max_sensor_range: float) -> float:
        return 1 / max_sensor_range 
    


def test_range_finder_model():
    meas_model = RangeFinderModel(w_hit=0.5, w_short=0.2, w_max=0.2, w_rand=0.1, lambda_short=0.1)

    # Define test data
    r = 0.3
    r_pred = 7.9

    likelihood = meas_model.correct_range_likelihood(r=r, r_pred=r_pred)

    print(f"\nLikelihood = {likelihood: .4f}")





def main():
    # rank_model_probs()
    # test()
    test_range_finder_model()


if __name__=="__main__":
    main()