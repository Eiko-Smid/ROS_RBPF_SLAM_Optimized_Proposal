
from typing import Optional, List, Tuple

from dataclasses import dataclass

import numpy as np

from ..infrastructure.defs import Pose2D
from geometry_msgs.msg import Pose2D as Pose2DMsg



@dataclass
class StepResult:
    # General metrics
    step_idx: Optional[int] = None
    t: Optional[float] = None
    t_step_duration: Optional[float] = None
    t_filter_duration: Optional[float] = None
    msg_queue_size: Optional[int] = None

    # Ground truth and raw odom
    true_pose: Optional[np.ndarray] = None
    raw_odom_pose: Optional[np.ndarray] = None

    # Scan matcher info
    scan_match_failed: Optional[bool] = None
    scan_match_fallback_failed: Optional[bool] = None

    # Particle poses before and after resampling
    particle_poses: Optional[np.ndarray] = None
    particle_weights: Optional[np.ndarray] = None
    particle_poses_before_resampling: Optional[np.ndarray] = None
    particle_weights_before_resampling: Optional[np.ndarray] = None

    # Weighted mean pose
    weighted_mean_pose: Optional[np.ndarray] = None

    # Best particle pose and weight
    max_weight_idx: Optional[int] = None
    best_particle_pose: Optional[np.ndarray] = None
    best_particle_weight: Optional[float] = None

    neff: Optional[float] = None
    resampling: Optional[bool] = None



class RBPFEvaluator:
    @staticmethod
    def _finite_values(values: List[Optional[float]]) -> np.ndarray:
        '''
        Converts the given list to a numpy array. Filters all non-finite values (inf, -inf, nan) and returns 
        the numpy array.
        '''
        arr = np.asarray(values, dtype=float)        
        return arr[np.isfinite(arr)]


    @staticmethod
    def _has_valid_metric_array(values: Optional[np.ndarray], min_size: int = 1) -> bool:
        """
        Returns True only if values is a numpy array with at least min_size finite entries.
        """
        if values is None or not isinstance(values, np.ndarray):
            return False
        if values.size < min_size:
            return False
        return bool(np.all(np.isfinite(values)))


    @staticmethod
    def _safe_mean(values: Optional[np.ndarray], min_size: int = 1, default: float = float("nan")) -> float:
        """
        Computes the mean only when a valid timing array is available; otherwise returns default.
        """
        if not RBPFEvaluator._has_valid_metric_array(values, min_size=min_size):
            return default
        return float(np.mean(values))

    
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


    def _weight_to_np_array(self, weights: List[float]) -> np.ndarray:
        """
        Converts a list of weights to a numpy array.
        """
        if weights is None:
            return None

        return np.array(weights, dtype=np.float64)

        
    @staticmethod
    def norm_weights(weights: np.ndarray) -> np.ndarray:
        """
        Normalizes the weights to sum to 1.
        """
        weights = np.asarray(weights, dtype=float)
        weight_sum = np.sum(weights)

        # if weight_sum <= 0:
        #     raise ValueError("Sum of weights is zero or negative, cannot normalize.")
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            raise ValueError(
                f"Invalid particle-weight sum: {weight_sum}"
            )

        return weights / weight_sum


    def _poses_to_np_array(self, poses: List[Pose2D]) -> np.ndarray:
        """
        Converts a list of pose tuples to a numpy array.
        """
        if poses is None:
            return None

        return np.array([self._pose_to_np_array(p) for p in poses], dtype=np.float64)


    @staticmethod
    def angle_diff(a: float, b: float) -> float:
        """
        Returns wrapped angular difference in [-pi, pi].
        """
        return np.arctan2(np.sin(a - b), np.cos(a - b))


    @staticmethod
    def translation_error(p1: Pose2D, p2: Pose2D) -> float:
        """
        Euclidean translation error in the x-y plane.
        """
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


    @staticmethod
    def weighted_mean_position(particle_poses: np.ndarray, particle_weights: np.ndarray) -> np.ndarray:
        '''
        Gets the particle poses and the corresponding weights and computes the weighted mean pose of all particles. 
        The x, y position is computed by weighting the position by its weights and then computing the mean. 
        
        The weighted mean orientation is computed by computing the weighted mean of the sin and cos of the angles. 
        This ensures that the mean orientation is computed correctly even if the angles wrap around 2*pi.
        '''
        # Convert to numpy array
        particle_poses = np.asarray(particle_poses, dtype=float)
        particle_weights = np.asarray(particle_weights, dtype=float)

        # Check dimensions, shape and finite values
        if particle_poses.ndim != 2 or particle_poses.shape[1] != 3:
            raise ValueError(
                f"Particle poses must have shape (N, 3) got {particle_poses.shape}"
            )
        
        if particle_weights.ndim != 1 or particle_weights.shape[0] != particle_poses.shape[0]:
            raise ValueError(
                f"Particle weights must have shape (N, ), where N is number of given poses"
                f"Number of given poses: {particle_poses.shape[0]} != number of weights: {particle_weights.shape[0]}"
            )
        
        if not np.all(np.isfinite(particle_poses)):
            raise ValueError(
                f"Particle poses contain non-finite values: {particle_poses}"
            )

        if not np.all(np.isfinite(particle_weights)):
            raise ValueError(
                f"Particle weights contain non-finite values: {particle_weights}"
            )

        # Normalize weights
        particle_weights = RBPFEvaluator.norm_weights(particle_weights)

        # Compute weighted position
        weighted_position = np.average(particle_poses[:, :2], axis=0, weights=particle_weights,)

        # Compute weighted orientation
        weighted_mean_rot = np.arctan2(
            np.sum(particle_weights * np.sin(particle_poses[:, 2])),
            np.sum(particle_weights * np.cos(particle_poses[:, 2]))
        )
        
        weighted_mean_poses = np.array([weighted_position[0], weighted_position[1], weighted_mean_rot])
        return weighted_mean_poses


    def evaluate_step(
        self,

        step_idx: Optional[int]=None,
        t: Optional[float]=None,
        step_duration: Optional[float]=None,
        filter_duration: Optional[float]=None,
        msg_queue_size: Optional[int]=None,

        scan_match_failed: Optional[bool]=None,
        scan_match_fallback_failed: Optional[bool]=None,

        raw_odom_pose: Optional[Pose2D]=None,
        particle_poses: List[Pose2D]=None,
        particle_weights: List[float]=None,

        particle_poses_before_resampling: List[Pose2D]=None,
        particle_weights_before_resampling: List[float]=None,

        neff: Optional[float]=None,
        particle_inherit_indices: Optional[List[int]]=None,

        true_pose: Optional[Pose2DMsg] = None,
    ) -> StepResult:        
        # Convert input data to numpy arrays
        particle_poses = self._poses_to_np_array(particle_poses)
        particle_weights = self._weight_to_np_array(particle_weights)
        particle_poses_before_resampling = self._poses_to_np_array(particle_poses_before_resampling)
        particle_weights_before_resampling = self._weight_to_np_array(particle_weights_before_resampling)

        # Convert raw odom to numpy array
        raw_odom_pose = self._pose_to_np_array(raw_odom_pose) if raw_odom_pose is not None else None

        # Convert true pose to numpy array
        if true_pose is not None:
            x = true_pose.x
            y = true_pose.y
            theta = true_pose.theta
            true_pose = np.array([x, y, theta], dtype=np.float64)

        # Init step results with None values
        step_res = StepResult()

        # Add general info to step
        step_res.step_idx = step_idx
        step_res.t = t
        step_res.t_step_duration = step_duration
        step_res.t_filter_duration = filter_duration
        step_res.msg_queue_size = msg_queue_size

        # Add scan matcher info
        step_res.scan_match_failed = bool(scan_match_failed) if scan_match_failed is not None else None 
        step_res.scan_match_fallback_failed = bool(scan_match_fallback_failed) if scan_match_fallback_failed is not None else None

        # Add particle poses
        step_res.particle_poses = particle_poses
        step_res.particle_weights = particle_weights
        step_res.particle_poses_before_resampling = particle_poses_before_resampling
        step_res.particle_weights_before_resampling = particle_weights_before_resampling
        step_res.true_pose = true_pose
        step_res.raw_odom_pose = raw_odom_pose

        # Compute weighted mean pose
        if particle_poses_before_resampling is not None and particle_weights_before_resampling is not None:
            step_res.weighted_mean_pose = self.weighted_mean_position(
                particle_poses=particle_poses_before_resampling,
                particle_weights=particle_weights_before_resampling,
            )

        # Compute best particle pose
        if particle_poses_before_resampling is not None and particle_weights_before_resampling is not None:
            step_res.max_weight_idx = np.argmax(particle_weights_before_resampling)
            step_res.best_particle_pose = particle_poses_before_resampling[step_res.max_weight_idx]
            step_res.best_particle_weight = particle_weights_before_resampling[step_res.max_weight_idx]

        # Resampling info
        step_res.neff = neff
        step_res.resampling = particle_inherit_indices is not None 
        
        return step_res

