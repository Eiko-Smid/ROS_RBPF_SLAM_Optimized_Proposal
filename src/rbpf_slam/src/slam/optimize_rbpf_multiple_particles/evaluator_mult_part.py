from dataclasses import dataclass, filed
from typing import Optional, Tuple, Dict, List

import numpy as np
from scipy.stats import spearmanr

from .playback_defs import ExperimentParams


# Transfers the angle in rad into meters to combine translational and rotational errors 
ROT_SCALE = 2.0     # trans_err + ROT_SCALE * angle (rad) -> m

Pose2D = Tuple[float, float, float]


class RBPFEValMultParticles:
    def __init__(self, rbpf):
        self.rbpf = rbpf

    def evaluate(self, particles):
        # Implement the evaluation logic for multiple particles
        # This is a placeholder for the actual evaluation code
        results = []
        for particle in particles:
            result = self.rbpf.evaluate_particle(particle)
            results.append(result)
        return results
    

    @staticmethod
    def _to_pose_tuple(pose) -> Optional[Pose2D]:
        """
        Converts a pose object to (x, y, theta).

        Supports tuples/lists/ndarrays and objects exposing x/y/theta attributes.
        """
        if pose is None:
            return None

        if hasattr(pose, "x") and hasattr(pose, "y") and hasattr(pose, "theta"):
            return (float(pose.x), float(pose.y), float(pose.theta))

        if isinstance(pose, (tuple, list, np.ndarray)) and len(pose) >= 3:
            return (float(pose[0]), float(pose[1]), float(pose[2]))

        raise TypeError(f"Unsupported pose format: {type(pose)}")
    

    def _pose_to_np_array(self, pose: Pose2D) -> np.ndarray:
        """
        Converts a pose tuple to a numpy array.
        """
        if pose is None:
            return None

        if hasattr(pose, "x") and hasattr(pose, "y") and hasattr(pose, "theta"):
            return np.array([pose.x, pose.y, pose.theta], dtype=np.float64)
        
        if isinstance(pose, (tuple, list, np.ndarray)) and len(pose) >= 3:
            return np.array(pose[:3], dtype=np.float64)

        raise TypeError(f"[Evaluator mp] Unsupported pose format: {type(pose)}")
        


    @staticmethod
    def angle_diff(a: float, b: float) -> float:
        """
        Returns wrapped angular difference in [-pi, pi].
        """
        return np.atan2(np.sin(a - b), np.cos(a - b))


    @staticmethod
    def translation_error(p1: Pose2D, p2: Pose2D) -> float:
        """
        Euclidean translation error in the x-y plane.
        """
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


    @staticmethod
    def pose_err(trans_err: float, rot_err: float, rot_scale: float) -> float:
        '''
        Get's the translational and rotational errors between two poses and computes a combined error metric
        that allows to compare the overall error of two poses.
        '''
        return float(np.sqrt(trans_err**2 + (rot_scale * rot_err)**2))


    def evaluate_step(
        self,
        step_idx: int,
        t: float,
        true_pose: Pose2D,
        raw_odom_pose: Pose2D,
        particle_poses: List[Pose2D],
        particle_weights: List[float],
        particle_inherit_indices: Optional[List[int]] = None
    ):
        '''
        Evaluates the performance of the RBPF at a given step.

        Ensure that the particle weights contain valid values. Also ensure that the poses contain valid values. 

        Parameters
        ----------
        step_idx : int
            The index of the current step in the experiment.
        t : float
            The timestamp of the current step.
        true_pose : Pose2D
            The ground truth pose of the robot at the current step.
        raw_odom_pose : Pose2D
            The raw odometry pose of the robot at the current step.
        particle_poses : List[Pose2D]
            The list of poses for each particle at the current step before resampling.
        particle_weights : List[float]
            The list of weights for each particle at the current step before resampling.
        '''
        # Convert poses to np arrays
        true_pose_arr = self._pose_to_np_array(true_pose)
        raw_odom_pose_arr = self._pose_to_np_array(raw_odom_pose)
        particle_poses_arr = [self._pose_to_np_array(p) for p in particle_poses]

        # Check for NaN values in poses
        if np.isnan(true_pose_arr).any():
            raise ValueError(f"[Evaluator mp] True pose contains NaN values at step {step_idx}: {true_pose_arr}")
        if np.isnan(raw_odom_pose_arr).any():
            raise ValueError(f"[Evaluator mp] Raw odometry pose contains NaN values at step {step_idx}: {raw_odom_pose_arr}")
        if any(np.isnan(p) for p in particle_poses_arr):
            raise ValueError(f"[Evaluator mp] Particle poses contain NaN values at step {step_idx}: {particle_poses_arr}")
        
        # Convert weights to np array
        if particle_weights is None or len(particle_weights) == 0:
            raise ValueError(f"[Evaluator mp] Particle weights are None or empty at step {step_idx}.")
        
        particle_weights_arr = np.array(particle_weights, dtype=np.float64)

        # Convert inherit indices to np array if provided
        if particle_inherit_indices is not None:
            raise ValueError(f"[Evaluator mp] Particle inherit indices should be None at step {step_idx}, but got: {particle_inherit_indices}")
        
        particle_inherit_indices_arr = np.array(particle_inherit_indices, dtype=np.int32)

        # Check for NaN values in particle weights
        if np.isnan(particle_weights_arr).any():
            raise ValueError(f"[Evaluator mp] Particle weights contain NaN values at step {step_idx}: {particle_weights_arr}")

        
        # 