#!/usr/bin/env python3
from dataclasses import dataclass

from math import atan2, cos, sin
import numpy as np
import time


@dataclass
class Pose2D:
    x: float
    y: float
    theta: float


def motion_probability_batch(
    x_new: np.ndarray,
    x_prev: np.ndarray,
    sigma_xy: float,
    sigma_theta: float,
) -> np.ndarray:
    """
    x_new shape: (N, 3)
    x_prev shape: (3,)
    """

    dx = x_new[:, 0] - x_prev[0]
    dy = x_new[:, 1] - x_prev[1]

    dtheta = x_new[:, 2] - x_prev[2]
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))

    return np.exp(
        -0.5 * (
            (dx / sigma_xy) ** 2 +
            (dy / sigma_xy) ** 2 +
            (dtheta / sigma_theta) ** 2
        )
    )


def motion_probability(x_new: Pose2D, x_prev: Pose2D, sigma_xy: float, sigma_theta: float) -> float:
    '''
    Get's the new pose and the previous pose and computes the motion probability based on the difference
    between the two poses and the noise parameters of the motion model.

    Parameters
    ----------
    x_new: Pose2D
        The new pose of the robot, given as a tuple (x, y, theta).
    x_prev: Pose2D
        The previous pose of the robot, given as a tuple (x, y, theta).
    
    Returns
    -------
    float: 
        The motion probability. The closer the poses are to each other, the higher the probability. 
    '''
    dx = x_new[0] - x_prev[0]
    dy = x_new[1] - x_prev[1]
    dtheta = x_new[2] - x_prev[2]
    dtheta = atan2(sin(dtheta), cos(dtheta))
    
    return np.exp(
    -0.5 * (
        (dx / sigma_xy) ** 2 +
        (dy / sigma_xy) ** 2 +
        (dtheta / sigma_theta) ** 2
    )
)


def test_motion_probability_batch():
    n_samples = 27
    new_poses = np.array([[1.0, 2.0, 0.5]] * n_samples)
    prev_pose = np.array([0.5, 1.5, 0.3])

    print("shape new_poses:", new_poses.shape)
    print("shape prev_pose:", prev_pose.shape)
    sigma_xy = 0.2
    sigma_theta = 0.15
    

    start_time = time.perf_counter()

    probs = motion_probability_batch(
        x_new=new_poses,
        x_prev=prev_pose,
        sigma_xy=sigma_xy,
        sigma_theta=sigma_theta,
    )

    end_time = time.perf_counter()
    mean_elapsed_time = ((end_time - start_time) * 1000.0) 
    print(f"Generated {n_samples} samples in {mean_elapsed_time:.4f} ms")



def test_old_motion_model():
    n_samples = 27
    new_pose = np.array([1.0, 2.0, 0.5])
    prev_pose = np.array([0.5, 1.5, 0.3])
    sigma_xy = 0.2
    sigma_theta = 0.15
    

    start_time = time.perf_counter()

    for i in range(n_samples):
        prob = motion_probability(
            x_new=new_pose,
            x_prev=prev_pose,
            sigma_xy=sigma_xy,
            sigma_theta=sigma_theta,
        )

    end_time = time.perf_counter()
    mean_elapsed_time = ((end_time - start_time) * 1000.0) 
    print(f"Generated {n_samples} samples in {mean_elapsed_time:.4f} ms")




def rank_xj_weights():
    xj_trans_errors_true = np.array([3.0, 2.0, 1.0]) 
    xj_weights = np.array([0.2, 0.3, 0.5])
    # xj_weights = np.array([0.5, 0.3, 0.2])

    # idx = 2
    idx_closest_true = np.argmin(xj_trans_errors_true)
    
    # [0.5, 0.3, 0.2]
    # - xj_weights = [-0.5, -0.3, -0.2]
    # argsort -> [0, 1, 2]
    order = np.argsort(-xj_weights)

    # print(np.where(order == idx_closest_true)[0][0])

    rank_of_closest = int(np.where(order == idx_closest_true)[0][0]) + 1

    norm = xj_weights.shape[0]
    rank_of_closest_norm = rank_of_closest / norm

    print(f"\n\nRank of closest xj is: {rank_of_closest}")
    print(f"\nRank of closest xj norm: {rank_of_closest_norm}")


def rank_xj_weights_optm(xj_trans_errors_true, xj_weights):
    idx_closest_true = np.argmin(xj_trans_errors_true)

    order = np.argsort(-xj_weights)

    rank_of_closest = int(np.where(order == idx_closest_true)[0][0]) + 1

    N = len(xj_weights)

    if N == 1:
        rank_score = 1.0
    else:
        rank_score = 1.0 - (rank_of_closest - 1) / (N - 1)

    print("idx_closest_true:", idx_closest_true)
    print("rank_of_closest:", rank_of_closest)
    print("rank_score:", rank_score)


def test_rank_xj_weights_optm():
    xj_trans_errors_true = np.array([0.1, 0.2, 0.3])
    xj_weights = np.array([0.3, 0.2, 0.1])
    rank_xj_weights_optm(xj_trans_errors_true, xj_weights)

    xj_trans_errors_true = np.array([0.2, 0.1, 0.3])
    xj_weights = np.array([0.3, 0.2, 0.1])
    rank_xj_weights_optm(xj_trans_errors_true, xj_weights)

    xj_trans_errors_true = np.array([0.2, 0.3, 0.1])
    xj_weights = np.array([0.3, 0.2, 0.1])
    rank_xj_weights_optm(xj_trans_errors_true, xj_weights)

    xj_trans_errors_true = np.array([0.2, 0.3, 0.1])
    xj_weights = np.array([0.2, 0.3, 0.1])
    rank_xj_weights_optm(xj_trans_errors_true, xj_weights)



def test():
    # xj_trans_errors_true = np.array([3.0, 2.0, 1.0])
    xj_trans_errors_true = np.array([1.0, 2.0, 3.0])
    xj_weights = np.array([0.2, 0.3, 0.5])

    idx_closest_true = np.argmin(xj_trans_errors_true)
    order = np.argsort(-xj_weights)

    rank_of_closest = np.where(order == idx_closest_true) 

    print(f"idx_closest_true: {idx_closest_true}")
    print(f"order: {order}")
    print(f"Rank of closest xj: {rank_of_closest}")





def main():

    # test_old_motion_model() 
    # test_motion_probability_batch()
    # rank_xj_weights()
    test_rank_xj_weights_optm()
    # test()


if __name__=="__main__":
    main()