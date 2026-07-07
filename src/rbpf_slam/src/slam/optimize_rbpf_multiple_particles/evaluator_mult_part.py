from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

import numpy as np
from scipy.stats import spearmanr

from .playback_defs import ExperimentParams


# Transfers the angle in rad into meters to combine translational and rotational errors 
ROT_SCALE = 2.0     # trans_err + ROT_SCALE * angle (rad) -> m

Pose2D = Tuple[float, float, float]


@dataclass
class StepResult:
    # General metrics
    step_idx: int
    t: float

    # Ground truth and odom
    true_pose: Pose2D
    raw_odom_pose: Pose2D

    # Particle poses before and after resampling
    particle_poses: np.ndarray
    particle_weights: np.ndarray
    particle_poses_before_resampling: np.ndarray
    particle_weights_before_resampling: np.ndarray

    # Weighted mean pose
    weighted_mean_pose: np.ndarray
    trans_err_weighted_mean: float
    rot_err_weighted_mean: float

    # Best particle pose and weight
    max_weight_idx: int
    best_particle_pose: np.ndarray
    best_particle_weight: float
    trans_err_best_particle: float
    rot_err_best_particle: float
    
    # Closest particle trajectory evaluation
    trans_errs_before_resampling: np.ndarray
    rot_errs_before_resampling: np.ndarray
    trans_err_closest_p_before_resampling: float
    rot_err_closest_p_before_resampling: float
    idx_closest_p_before_resampling: int

    trans_errs_after_resampling: np.ndarray
    rot_errs_after_resampling: np.ndarray
    trans_err_closest_p_after_resampling: float
    rot_err_closest_p_after_resampling: float
    idx_closest_p_after_resampling: int
    
    particle_inherit_indices: Optional[np.ndarray] = None
    

@dataclass
class RunResult:
    """
    Stores all RBPF evaluation data for one parameter-set run.
    """
    params: ExperimentParams
    step_results: List[StepResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)



class RBPFEValMultParticles:

    @staticmethod
    def _finite_values(values: List[Optional[float]]) -> np.ndarray:
        '''
        Converts the given list to a numpy array. Filters all non-finite values (inf, -inf, nan) and returns 
        the numpy array.
        '''
        arr = np.asarray(values, dtype=float)        
        return arr[np.isfinite(arr)]


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
    def pose_err(trans_err: float, rot_err: float, rot_scale: float) -> float:
        '''
        Get's the translational and rotational errors between two poses and computes a combined error metric
        that allows to compare the overall error of two poses.
        '''
        return float(np.sqrt(trans_err**2 + (rot_scale * rot_err)**2))
    

    @staticmethod
    def norm_weights(weights: np.ndarray) -> np.ndarray:
        """
        Normalizes the weights to sum to 1.
        """
        weights = np.asarray(weights, dtype=float)
        weight_sum = np.sum(weights)
        
        if weight_sum <= 0:
            raise ValueError("Sum of weights is zero or negative, cannot normalize.")
        
        return weights / weight_sum


    @staticmethod
    def weighted_mean_position(particle_poses: np.ndarray, particle_weights: np.ndarray) -> np.ndarray:
        '''
        Gets the particle poses and the corresponding weights and computes the weighted mean pose of all particles. 
        The x, y postion is computed by weighted the postion by its weights and the computing the mean. 
        
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
        particle_weights = RBPFEValMultParticles.norm_weights(particle_weights)

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
        step_idx: int,
        t: float,
        true_pose: Pose2D,
        raw_odom_pose: Pose2D,
        particle_poses: List[Pose2D],
        particle_weights: List[float],

        particle_poses_before_resampling: List[Pose2D],
        particle_weights_before_resampling: List[float],

        particle_inherit_indices: Optional[List[int]] = None
    ) -> StepResult:
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

        Returns
        -------
        StepResult
            A dataclass containing the evaluation results for the current step.
        '''
        '''
        TODO:
        1) Add step duration
        2) Add scan match failure info
        3) Optional: Add measurement model metrics
        '''

        # Convert poses to np arrays
        true_pose_arr = self._pose_to_np_array(true_pose)
        raw_odom_pose_arr = self._pose_to_np_array(raw_odom_pose)
        particle_poses_arr = self._poses_to_np_array(particle_poses)

        particle_poses_before_resampling_arr = self._poses_to_np_array(particle_poses_before_resampling)
        particle_weights_before_resampling_arr = np.array(particle_weights_before_resampling, dtype=np.float64)

        # Store indices of particles if resampling too place, otherwise None
        particle_inherit_indices_arr = np.array(particle_inherit_indices, dtype=np.int32) if particle_inherit_indices is not None else None

        # Check for none finite
        if not np.all(np.isfinite(true_pose_arr)):
            raise ValueError(f"[Evaluator mp] True pose contains non-finite values at step {step_idx}: {true_pose_arr}")
        if not np.all(np.isfinite(raw_odom_pose_arr)):
            raise ValueError(f"[Evaluator mp] Raw odometry pose contains non-finite values at step {step_idx}: {raw_odom_pose_arr}")
        if not np.all(np.isfinite(particle_poses_arr)):
            raise ValueError(f"[Evaluator mp] Particle poses contain non-finite values at step {step_idx}: {particle_poses_arr}")
        if not np.all(np.isfinite(particle_weights_before_resampling_arr)):
            raise ValueError(f"[Evaluator mp] Particle weights before resampling contain non-finite values at step {step_idx}: {particle_weights_before_resampling_arr}")
        if not np.all(np.isfinite(particle_poses_before_resampling_arr)):
            raise ValueError(f"[Evaluator mp] Particle poses before resampling contain non-finite values at step {step_idx}: {particle_poses_before_resampling_arr}")
        
        # Convert weights to np array
        if particle_weights is None or len(particle_weights) == 0:
            raise ValueError(f"[Evaluator mp] Particle weights are None or empty at step {step_idx}.")
        
        particle_weights_arr = np.array(particle_weights, dtype=np.float64)

        # Convert inherit indices to np array if provided
        # if particle_inherit_indices is not None:
        #     raise ValueError(f"[Evaluator mp] Particle inherit indices should be None at step {step_idx}, but got: {particle_inherit_indices}")

        # Check for NaN values in particle weights
        if np.isnan(particle_weights_arr).any():
            raise ValueError(f"[Evaluator mp] Particle weights contain NaN values at step {step_idx}: {particle_weights_arr}")

        # Normalize weights
        weight_sum = np.sum(particle_weights_arr)
        
        if weight_sum <= 0:
            raise ValueError(f"[Evaluator mp] Sum of particle weights is zero or negative at step {step_idx}, cannot compute weighted mean.")
        
        particle_weights_arr = self.norm_weights(particle_weights_arr)

        # Compute weighted mean poses
        weighted_mean_pose = self.weighted_mean_position(particle_poses_arr, particle_weights_arr)
        # Compute trans and rot error of weighted mean pose
        trans_err_weighted_mean = self.translation_error(weighted_mean_pose, true_pose_arr)
        rot_err_weighted_mean = np.abs(self.angle_diff(weighted_mean_pose[2], true_pose_arr[2]))

        # Estimate particle with highest weight -> Will give us online trajectory in summary
        max_weight_idx = np.argmax(particle_weights_before_resampling_arr)
        best_particle_pose = particle_poses_before_resampling_arr[max_weight_idx]
        best_particle_weight = particle_weights_before_resampling_arr[max_weight_idx]
        # Compute trans and rot error of particle with highest weight
        trans_err_best_particle = self.translation_error(best_particle_pose, true_pose_arr)
        rot_err_best_particle = np.abs(self.angle_diff(best_particle_pose[2], true_pose_arr[2]))

        # Estimate closest particle before resampling -> Will give us offline trajectory in summary
        xy_diffs_before_resampling = particle_poses_before_resampling_arr[:, :2] - true_pose_arr[:2]
        trans_errors_before_resampling = np.linalg.norm(xy_diffs_before_resampling, axis=1)

        rot_diffs_before_resampling = particle_poses_before_resampling_arr[:, 2] - true_pose_arr[2]
        rot_errs_before_resampling = np.abs(
            np.arctan2(
                np.sin(rot_diffs_before_resampling),
                np.cos(rot_diffs_before_resampling),
            )
        )

        # Find closest particle
        idx_closest_p_before_resampling = np.argmin(trans_errors_before_resampling)
        trans_err_closest_p_before_resampling = trans_errors_before_resampling[idx_closest_p_before_resampling]
        rot_err_closest_p_before_resampling = rot_errs_before_resampling[idx_closest_p_before_resampling]

        # Estimate closest particle after resampling
        xy_diffs_after_resampling = particle_poses_arr[:, :2] - true_pose_arr[:2]
        trans_errors_after_resampling = np.linalg.norm(xy_diffs_after_resampling, axis=1)

        rot_diffs_after_resampling = particle_poses_arr[:, 2] - true_pose_arr[2]
        rot_errs_after_resampling = np.abs(
            np.arctan2(
                np.sin(rot_diffs_after_resampling),
                np.cos(rot_diffs_after_resampling),
            )
        )

        # Find closest particle after resampling
        idx_closest_p_after_resampling = np.argmin(trans_errors_after_resampling)
        trans_err_closest_p_after_resampling = trans_errors_after_resampling[idx_closest_p_after_resampling]
        rot_err_closest_p_after_resampling = rot_errs_after_resampling[idx_closest_p_after_resampling]

        # Store results
        return StepResult(
            # General metrics
            step_idx=step_idx,
            t=t,

            # Ground truth and odom
            true_pose=true_pose_arr,
            raw_odom_pose=raw_odom_pose_arr,

            # Particle poses before and after resampling
            particle_poses=particle_poses_arr,
            particle_weights=particle_weights_arr,
            particle_poses_before_resampling=particle_poses_before_resampling_arr,
            particle_weights_before_resampling=particle_weights_before_resampling_arr,
            
            # Weighted mean pose
            weighted_mean_pose=weighted_mean_pose,
            trans_err_weighted_mean=trans_err_weighted_mean,
            rot_err_weighted_mean=rot_err_weighted_mean,
            
            # Best particle pose and weight
            max_weight_idx=max_weight_idx,
            best_particle_pose=best_particle_pose,
            best_particle_weight=best_particle_weight,
            trans_err_best_particle=trans_err_best_particle,
            rot_err_best_particle=rot_err_best_particle,                            

            # Closest particle trajectory evaluation
            trans_errs_before_resampling=trans_errors_before_resampling,
            rot_errs_before_resampling=rot_errs_before_resampling,
            trans_err_closest_p_before_resampling=trans_err_closest_p_before_resampling,
            rot_err_closest_p_before_resampling=rot_err_closest_p_before_resampling,
            idx_closest_p_before_resampling=idx_closest_p_before_resampling,

            trans_errs_after_resampling=trans_errors_after_resampling,
            rot_errs_after_resampling=rot_errs_after_resampling,            
            trans_err_closest_p_after_resampling=trans_err_closest_p_after_resampling,
            rot_err_closest_p_after_resampling=rot_err_closest_p_after_resampling,            
            idx_closest_p_after_resampling=idx_closest_p_after_resampling,

            # Optional particle inherit indices (if resampling took place) -> Recover trajectory of best particle at the end
            particle_inherit_indices=particle_inherit_indices_arr,            
        )


    def _restore_map_trajectory_errors(
        self,
        best_particle_idx: int,
        particle_inherit_indices: List[Optional[np.ndarray]],
        trans_errs_before_resampling_list: List[np.ndarray],
        rot_errs_before_resampling_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Restore the pre-resampling error trajectory of the final MAP particle. The given particle_inherit_indices list 
        contains the indices of the parent particles and None if no resampling took place. With this list the trajectory
        of the best particle at the end of the run will be restored. 

        Returns the translational and rotational errors of the MAP trajectory in chronological order. Also cleans the 
        output by estimating the valid values, that the trans and rot error arrays have in common.

        Parameters
        ----------
        best_particle_idx : int
            The index of the best particle at the last step of the run.
        particle_inherit_indices : List[Optional[np.ndarray]]
            A list of arrays containing the indices of the parent particles for each step. If no resampling took place at 
            a step, the corresponding entry is None.  
        trans_errs_before_resampling_list : List[np.ndarray]
            A list of arrays containing the translational errors before resampling for each step.
        rot_errs_before_resampling_list : List[np.ndarray]
            A list of arrays containing the rotational errors before resampling for each step.
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Two numpy arrays containing the cleaned translational and rotational errors of the MAP trajectory in chronological
            order.
        """
        # Initialize index with best particle index in last iteration of the filter
        idx = int(best_particle_idx)

        trans_errs = []
        rot_errs = []

        # Walk backward through the run.
        for step_idx in reversed(range(len(trans_errs_before_resampling_list))):

            # Store errors of current particle 
            trans_errs.append(trans_errs_before_resampling_list[step_idx][idx])
            rot_errs.append(rot_errs_before_resampling_list[step_idx][idx])

            # Skip this in first iteration
            if step_idx > 0:
                # Get the index of the parent particle if available
                indices = particle_inherit_indices[step_idx - 1]

                # If resampling took place, update the index to the parent particle's index
                if indices is not None:
                    idx = int(indices[idx])

        # We collected errors backward, so reverse to chronological order.
        trans_errs = np.asarray(trans_errs[::-1], dtype=float)
        rot_errs = np.asarray(rot_errs[::-1], dtype=float)

        # Exclude non values
        valid_mask = np.isfinite(trans_errs) & np.isfinite(rot_errs)
        trans_errs = trans_errs[valid_mask]
        rot_errs = rot_errs[valid_mask]

        return trans_errs, rot_errs



    def summarize_run(self, step_results: List[StepResult], params: Optional[ExperimentParams] = None) -> Dict:
        # Filter pose errors
        # Weighted mean pose errors
        trans_errs_weighted_mean = self._finite_values([step.trans_err_weighted_mean for step in step_results])
        rot_errs_weighted_mean = self._finite_values([step.rot_err_weighted_mean for step in step_results])

        # Best particle pose errors
        trans_errs_best_particle = self._finite_values([step.trans_err_best_particle for step in step_results])
        rot_errs_best_particle = self._finite_values([step.rot_err_best_particle for step in step_results])

        # Closest particle pose errors before resampling
        trans_errs_closest_p_before_resampling = self._finite_values(
            [step.trans_err_closest_p_before_resampling for step in step_results]
        )
        rot_errs_closest_p_before_resampling = self._finite_values(
            [step.rot_err_closest_p_before_resampling for step in step_results]
        )

        # Get trans errors before resampling
        trans_errs_before_resampling_list = [step.trans_errs_before_resampling for step in step_results]
        rot_errs_before_resampling_list = [step.rot_errs_before_resampling for step in step_results]

        # Restore MAP trajectory errors
        best_p_idx = step_results[-1].max_weight_idx
        particle_inherit_indices = [step.particle_inherit_indices for step in step_results]

        trans_errs_map_traj, rot_errs_map_traj = self._restore_map_trajectory_errors(
            best_particle_idx=best_p_idx,
            particle_inherit_indices=particle_inherit_indices,
            trans_errs_before_resampling_list=trans_errs_before_resampling_list,
            rot_errs_before_resampling_list=rot_errs_before_resampling_list
        )
        # if step_results:
        #     best_p_idx = step_results[-1].max_weight_idx
        #     particle_inherit_indices = [step.particle_inherit_indices for step in step_results]

        #     trans_errs_map_traj, rot_errs_map_traj = self._restore_map_trajectory_errors(
        #         best_particle_idx=best_p_idx,
        #         particle_inherit_indices=particle_inherit_indices,
        #         trans_errs_before_resampling_list=trans_errs_before_resampling_list,
        #         rot_errs_before_resampling_list=rot_errs_before_resampling_list
        #     )
        # else:
        #     trans_errs_map_traj = np.array([], dtype=float)
        #     rot_errs_map_traj = np.array([], dtype=float)

        has_weighted_mean_errors = (
            len(trans_errs_weighted_mean) > 0 and len(rot_errs_weighted_mean) > 0
        )
        has_best_particle_errors = (
            len(trans_errs_best_particle) > 0 and len(rot_errs_best_particle) > 0
        )
        has_closest_particle_before_resampling_errors = (
            len(trans_errs_closest_p_before_resampling) > 0
            and len(rot_errs_closest_p_before_resampling) > 0
        )
        has_map_trajectory_errors = (
            len(trans_errs_map_traj) > 0 and len(rot_errs_map_traj) > 0
        )

        summary = {
            "n_steps": len(step_results),

            # Weighted mean pose trajectory
            "mean_trans_err_weighted_mean": (
                float(np.mean(trans_errs_weighted_mean))
                if has_weighted_mean_errors else None
            ),
            "rmse_trans_err_weighted_mean": (
                float(np.sqrt(np.mean(np.square(trans_errs_weighted_mean))))
                if has_weighted_mean_errors else None
            ),
            "worst_trans_err_weighted_mean": (
                float(np.max(trans_errs_weighted_mean))
                if has_weighted_mean_errors else None
            ),
            "mean_rot_err_weighted_mean": (
                float(np.mean(rot_errs_weighted_mean))
                if has_weighted_mean_errors else None
            ),
            "rmse_rot_err_weighted_mean": (
                float(np.sqrt(np.mean(np.square(rot_errs_weighted_mean))))
                if has_weighted_mean_errors else None
            ),
            "worst_rot_err_weighted_mean": (
                float(np.max(rot_errs_weighted_mean))
                if has_weighted_mean_errors else None
            ),
            "final_trans_drift_trans_err_weighted_mean": (
                float(trans_errs_weighted_mean[-1])
                if has_weighted_mean_errors else None
            ),
            "final_rot_drift_rot_err_weighted_mean": (
                float(rot_errs_weighted_mean[-1])
                if has_weighted_mean_errors else None
            ),

            # Online MAP / highest-weight particle trajectory
            "mean_trans_err_best_particle": (
                float(np.mean(trans_errs_best_particle))
                if has_best_particle_errors else None
            ),
            "rmse_trans_err_best_particle": (
                float(np.sqrt(np.mean(np.square(trans_errs_best_particle))))
                if has_best_particle_errors else None
            ),
            "worst_trans_err_best_particle": (
                float(np.max(trans_errs_best_particle))
                if has_best_particle_errors else None
            ),
            "mean_rot_err_best_particle": (
                float(np.mean(rot_errs_best_particle))
                if has_best_particle_errors else None
            ),
            "rmse_rot_err_best_particle": (
                float(np.sqrt(np.mean(np.square(rot_errs_best_particle))))
                if has_best_particle_errors else None
            ),
            "worst_rot_err_best_particle": (
                float(np.max(rot_errs_best_particle))
                if has_best_particle_errors else None
            ),
            "final_trans_drift_trans_err_best_particle": (
                float(trans_errs_best_particle[-1])
                if has_best_particle_errors else None
            ),
            "final_rot_drift_rot_err_best_particle": (
                float(rot_errs_best_particle[-1])
                if has_best_particle_errors else None
            ),

            # Oracle closest-particle trajectory before resampling
            "mean_trans_err_closest_p_before_resampling": (
                float(np.mean(trans_errs_closest_p_before_resampling))
                if has_closest_particle_before_resampling_errors else None
            ),
            "rmse_trans_err_closest_p_before_resampling": (
                float(np.sqrt(np.mean(np.square(trans_errs_closest_p_before_resampling))))
                if has_closest_particle_before_resampling_errors else None
            ),
            "worst_trans_err_closest_p_before_resampling": (
                float(np.max(trans_errs_closest_p_before_resampling))
                if has_closest_particle_before_resampling_errors else None
            ),
            "mean_rot_err_closest_p_before_resampling": (
                float(np.mean(rot_errs_closest_p_before_resampling))
                if has_closest_particle_before_resampling_errors else None
            ),
            "rmse_rot_err_closest_p_before_resampling": (
                float(np.sqrt(np.mean(np.square(rot_errs_closest_p_before_resampling))))
                if has_closest_particle_before_resampling_errors else None
            ),
            "worst_rot_err_closest_p_before_resampling": (
                float(np.max(rot_errs_closest_p_before_resampling))
                if has_closest_particle_before_resampling_errors else None
            ),
            "final_trans_drift_trans_err_closest_p_before_resampling": (
                float(trans_errs_closest_p_before_resampling[-1])
                if has_closest_particle_before_resampling_errors else None
            ),
            "final_rot_drift_rot_err_closest_p_before_resampling": (
                float(rot_errs_closest_p_before_resampling[-1])
                if has_closest_particle_before_resampling_errors else None
            ),

            # Final MAP particle ancestral trajectory
            "mean_trans_err_map_traj": (
                float(np.mean(trans_errs_map_traj))
                if has_map_trajectory_errors else None
            ),
            "rmse_trans_err_map_traj": (
                float(np.sqrt(np.mean(np.square(trans_errs_map_traj))))
                if has_map_trajectory_errors else None
            ),
            "worst_trans_err_map_traj": (
                float(np.max(trans_errs_map_traj))
                if has_map_trajectory_errors else None
            ),
            "mean_rot_err_map_traj": (
                float(np.mean(rot_errs_map_traj))
                if has_map_trajectory_errors else None
            ),
            "rmse_rot_err_map_traj": (
                float(np.sqrt(np.mean(np.square(rot_errs_map_traj))))
                if has_map_trajectory_errors else None
            ),
            "worst_rot_err_map_traj": (
                float(np.max(rot_errs_map_traj))
                if has_map_trajectory_errors else None
            ),
            "final_trans_drift_trans_err_map_traj": (
                float(trans_errs_map_traj[-1])
                if has_map_trajectory_errors else None
            ),
            "final_rot_drift_rot_err_map_traj": (
                float(rot_errs_map_traj[-1])
                if has_map_trajectory_errors else None
            ),
        }

        return summary


def test():
    my_list = [np.array([1, 1]), np.array([2, 2]), np.array([3, 3]), np.array([4, 4])]

    # len(my_list) is 4. range(3) gives [0, 1, 2]. reversed gives [2, 1, 0].
    for i in reversed(range(len(my_list) - 1)):
        arr = my_list[i]
        print(f"Index i: {i}, Array: {arr}")


def main():
    test()


if __name__ == "__main__":
    main()
