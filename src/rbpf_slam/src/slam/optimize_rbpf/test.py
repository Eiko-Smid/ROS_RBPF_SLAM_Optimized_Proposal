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


def main():
    test_old_motion_model() 
    test_motion_probability_batch()


if __name__=="__main__":
    main()