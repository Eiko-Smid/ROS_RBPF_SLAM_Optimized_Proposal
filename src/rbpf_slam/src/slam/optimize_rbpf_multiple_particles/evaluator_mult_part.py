from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Sequence, Union

import numpy as np
from scipy.stats import spearmanr

from .playback_defs import ExperimentParams


X = 0
Y = 1
THETA = 2

# Transfers the angle in rad into meters to combine translational and rotational errors 
ROT_SCALE = 2.0     # trans_err + ROT_SCALE * angle (rad) -> m

# Threshold for best particle errors (higher -> no acceptable error)
TRANS_ERRS_BEST_PARTICLE_THRES = 0.1
ROT_ERRS_BEST_PARTICLE_THRES = np.radians(3.0)

# Threshold for map trajectory errors (higher -> no acceptable error)
TRANS_ERRS_MAP_TRAJ_THRES = 0.1
ROT_ERRS_MAP_TRAJ_THRES = np.radians(3.0)

# Define max allowed gap between best particle and particle closest to true pose
MAX_TRANS_GAP = 0.05
MAX_ROT_GAP = np.radians(1.5)

Pose2D = Tuple[float, float, float]


@dataclass
class StepResult:
    # General metrics
    step_idx: Optional[int] = None
    t: Optional[float] = None
    t_step_duration: Optional[float] = None

    # Ground truth and odom
    true_pose: Optional[np.ndarray] = None
    raw_odom_pose: Optional[np.ndarray] = None

    # Scan matcher info
    scan_match_failed: Optional[bool] = None
    scan_match_failed_fallback: Optional[bool] = None

    # Particle poses before and after resampling
    particle_poses: Optional[np.ndarray] = None
    particle_weights: Optional[np.ndarray] = None
    particle_poses_before_resampling: Optional[np.ndarray] = None
    particle_weights_before_resampling: Optional[np.ndarray] = None

    # Weighted mean pose
    weighted_mean_pose: Optional[np.ndarray] = None
    trans_err_weighted_mean: Optional[float] = None
    rot_err_weighted_mean: Optional[float] = None

    # Weighted particle standard deviation
    weighted_part_std_x: Optional[float] = None
    weighted_part_std_y: Optional[float] = None
    weighted_part_std_theta: Optional[float] = None
    weighted_part_std_pos: Optional[float] = None

    # Errors of raw odom
    trans_err_raw_odom: Optional[float] = None
    rot_err_raw_odom: Optional[float] = None

    # Improvement metrics
    trans_err_weighted_mean_impr_over_raw_odom: Optional[float] = None
    rot_err_weighted_mean_impr_over_raw_odom: Optional[float] = None

    # Best particle pose and weight
    max_weight_idx: Optional[int] = None
    best_particle_pose: Optional[np.ndarray] = None
    best_particle_weight: Optional[float] = None
    trans_err_best_particle: Optional[float] = None
    rot_err_best_particle: Optional[float] = None
    
    # Closest particle trajectory evaluation
    trans_errs_before_resampling: Optional[np.ndarray] = None
    rot_errs_before_resampling: Optional[np.ndarray] = None
    trans_err_closest_p_before_resampling: Optional[float] = None
    rot_err_closest_p_before_resampling: Optional[float] = None
    idx_closest_p_before_resampling: Optional[int] = None
    gap_trans_best_p_to_closest_before_resamp: Optional[float] = None
    gap_rot_best_p_to_closest_before_resamp: Optional[float] = None

    gap_trans_best_to_min_before_resamp: Optional[float] = None
    gap_rot_best_to_min_before_resamp: Optional[float] = None

    trans_errs_after_resampling: Optional[np.ndarray] = None
    rot_errs_after_resampling: Optional[np.ndarray] = None
    trans_err_closest_p_after_resampling: Optional[float] = None
    rot_err_closest_p_after_resampling: Optional[float] = None
    idx_closest_p_after_resampling: Optional[int] = None
    trans_closest_p_after_before_resamp: Optional[float] = None
    rot_closest_p_after_before_resamp: Optional[float] = None
    
    neff: Optional[float] = None
    neff_ratio: Optional[float] = None
    particle_inherit_indices: Optional[np.ndarray] = None
    resampling: Optional[bool] = None
    unique_parents: Optional[int] = None

    # Correlational metrics
    corr_trans_weights_pos: Optional[float] = None
    corr_rot_weights_pos: Optional[float] = None

    # Time metrics
    # Proposal time durations
    t_sample_poses: Optional[float] = None
    t_pred_poses: Optional[float] = None
    t_motion_model: Optional[float] = None
    t_meas_model: Optional[float] = None
    t_compute_prop_params: Optional[float] = None
    t_sample_from_prop: Optional[float] = None
    
    # Scan matcher time durations
    time_duration_scan_matching: Optional[float] = None
    time_duration_prediction: Optional[float] = None
    time_duration_map_extraction: Optional[float] = None
    time_duration_correct_pose: Optional[float] = None
    time_duration_update_pose: Optional[float] = None

    # ICP time durations
    t_init_icp_trans: Optional[float] = None
    t_init_and_train_nn_tree_normals: Optional[float] = None
    t_downsampling_pointcloud: Optional[float] = None
    t_compute_normal: Optional[float] = None
    t_outlier_rejection: Optional[float] = None
    t_find_nn_outlier_rejec: Optional[float] = None
    t_prepare_system: Optional[float] = None
    t_solve_least_squares: Optional[float] = None
    t_transf_update_and_results: Optional[float] = None
    t_find_trans: Optional[float] = None
    

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
        if not RBPFEValMultParticles._has_valid_metric_array(values, min_size=min_size):
            return default
        return float(np.mean(values))


    @staticmethod
    def _safe_spearman_correlation(a: np.ndarray, b: np.ndarray) -> Optional[float]:
        """
        Computes the Spearman correlation only for aligned, finite, non-constant arrays.
        """
        # Transform data to numpy arrays
        arr_a = np.asarray(a, dtype=float)
        arr_b = np.asarray(b, dtype=float)
        
        # Validate data shape and finite vals
        if arr_a.shape != arr_b.shape or arr_a.size < 2:
            return None

        finite_mask = np.isfinite(arr_a) & np.isfinite(arr_b)
        if np.count_nonzero(finite_mask) < 2:
            return None

        arr_a = arr_a[finite_mask]
        arr_b = arr_b[finite_mask]

        # Skip spearmen if values are to close to each other -> avoid numpy warnings in terminal
        if np.allclose(arr_a, arr_a[0]) or np.allclose(arr_b, arr_b[0]):
            return None

        corr = spearmanr(arr_a, arr_b).correlation
        if corr is None or not np.isfinite(corr):
            return None
        return float(corr)


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
    def trans_err_trajectory(traj_1: Sequence[Union[Pose2D, np.ndarray]], traj_2: Sequence[Union[Pose2D, np.ndarray]]) -> List[float]:
        """
        Computes the translational errors between two trajectories in a safe manner. The given trajectorys
        needs to be of same length, otherwise they will raise an error.

        Parameters
        ----------
        traj_1 : List[Pose2D]
            The first trajectory as a list of poses.
        traj_2 : List[Pose2D]
            The second trajectory as a list of poses.
        
        Returns
        -------
        np.ndarray
            The translational errors between the two trajectories.
        """
        # Convert trajectorys into numpy arrays
        traj_1 = np.asarray(traj_1, dtype=float)  
        traj_2 = np.asarray(traj_2, dtype=float)

        # Do comparability checks
        if traj_1.ndim != 2 or traj_2.ndim != 2:
            raise ValueError(
                f"Number of dimension of given traj differ!"
                f"ndim traj 1: {traj_1.ndim},\nndim traj 2: {traj_2.ndim}"
            )

        if traj_1.shape[1] < 2 or traj_2.shape[1] < 2:
            raise ValueError(
                f"Both trajectorys must have at least 2 columns (x, y)!"
                f"N columns traj_1: {traj_1.shape[1]}, \nN columns traj_2: {traj_2.shape[1]}"
            )
        
        if traj_1.shape[0] != traj_2.shape[0]:
            raise ValueError(
                f"Both trajectorys must have same number of rows!"
                f"N rows traj_1: {traj_1.shape[0]}, \nN rows traj_2: {traj_2.shape[0]}"
            )
        
        # Find union valid mask
        valid_mask = np.all(np.isfinite(traj_1) & np.isfinite(traj_2), axis=1)

        traj_1 = traj_1[valid_mask]
        traj_2 = traj_2[valid_mask]

        # Compute trans err
        trans_errs = np.linalg.norm(traj_1[:, :2] - traj_2[:, :2], axis=1)

        return trans_errs
        

    @staticmethod
    def rot_err_trajectory(traj_1: Sequence[Union[Pose2D, np.ndarray]], traj_2: Sequence[Union[Pose2D, np.ndarray]]) -> List[float]:
        """
        Computes the rotational errors between two trajectories in a safe manner. The given trajectorys
        needs to be of same length, otherwise they will raise an error.

        Parameters
        ----------
        traj_1 : Sequence[Union[Pose2D, np.ndarray]]
            The first trajectory as a sequence of poses.
        traj_2 : Sequence[Union[Pose2D, np.ndarray]]
            The second trajectory as a sequence of poses.
        
        Returns
        -------
        List[float]
            The absolute rotational errors between the two trajectories.
        """
        # Convert trajectorys into numpy arrays
        traj_1 = np.asarray(traj_1, dtype=float)  
        traj_2 = np.asarray(traj_2, dtype=float)

        # Do comparability checks
        if traj_1.ndim != 2 or traj_2.ndim != 2:
            raise ValueError(
                f"Number of dimension of given traj differ!"
                f"ndim traj 1: {traj_1.ndim},\nndim traj 2: {traj_2.ndim}"
            )

        if traj_1.shape[1] < 3 or traj_2.shape[1] < 3:
            raise ValueError(
                f"Both trajectorys must have at least 3 columns (x, y, theta)!"
                f"N columns traj_1: {traj_1.shape[1]}, \nN columns traj_2: {traj_2.shape[1]}"
            )
        
        if traj_1.shape[0] != traj_2.shape[0]:
            raise ValueError(
                f"Both trajectorys must have same number of rows!"
                f"N rows traj_1: {traj_1.shape[0]}, \nN rows traj_2: {traj_2.shape[0]}"
            )
        
        # Find union valid mask
        valid_mask = np.all(np.isfinite(traj_1) & np.isfinite(traj_2), axis=1)

        traj_1 = traj_1[valid_mask]
        traj_2 = traj_2[valid_mask]

        # Compute rot err
        rot_diffs = np.arctan2(
            np.sin(traj_1[:, 2] - traj_2[:, 2]),
            np.cos(traj_1[:, 2] - traj_2[:, 2])
        )

        rot_errs = np.abs(rot_diffs)

        return rot_errs


    @staticmethod
    def pose_err(trans_err: float, rot_err: float, rot_scale: float) -> float:
        '''
        Get's the translational and rotational errors between two poses and computes a combined error metric
        that allows to compare the overall error of two poses.
        '''
        return float(np.sqrt(trans_err**2 + (rot_scale * rot_err)**2))


    @staticmethod
    def compute_improvement(a, b, eps=1e-12):
        """
        Computes the improvement of a over b as (b - a) / (b + eps).
        """
        a = float(a)
        b = float(b)

        if not np.isfinite(a) or not np.isfinite(b):
            return float("nan")

        if abs(b) <= eps:
            return float("nan")

        improvement = (b - a) / (b + eps)
        if not np.isfinite(improvement):
            return float("nan")

        return float(improvement)
        

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
    

    def compute_weighted_part_std(self, particle_poses: np.ndarray, particle_weights: np.ndarray, weighted_mean_poses: np.ndarray):
        '''
        Compute weighted standard deviation of the particle poses.

        Parameters
        ----------
        particle_poses : np.ndarray
            Array of shape (N, 3) containing the poses of the particles.
        particle_weights : np.ndarray
            Array of shape (N,) containing the weights of the particles.
        weighted_mean_poses : np.ndarray
            Array of shape (3,) containing the weighted mean pose of the particles.

        Returns
        -------
        weighted_std_x : float
            Weighted standard deviation of the x positions.
        weighted_std_y : float
            Weighted standard deviation of the y positions.
        weighted_std_theta : float
            Weighted standard deviation of the orientations.
        weighted_pos_std : float
            Weighted standard deviation of the positions (x, y).
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
                f"Particle weights must have shape (N, ), where N is number of given poses. "
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

        # Compute weighted var translation
        dxs = (particle_poses[:, X] - weighted_mean_poses[X])
        dys = (particle_poses[:, Y] - weighted_mean_poses[Y])

        weighted_var_x = np.sum(particle_weights * dxs**2)
        weighted_var_y = np.sum(particle_weights * dys**2)
        
        # Compute weighted var rotation
        theta_diff = np.arctan2(
            np.sin(particle_poses[:, THETA] - weighted_mean_poses[THETA]),
            np.cos(particle_poses[:, THETA] - weighted_mean_poses[THETA]),
        )
    
        weighted_var_theta = np.sum(particle_weights * theta_diff**2)

        # Compute stds
        weighted_std_x = np.sqrt(weighted_var_x)
        weighted_std_y = np.sqrt(weighted_var_y)
        weighted_std_theta = np.sqrt(weighted_var_theta)

        # Compute weighted particle position
        weighted_pos_std = np.sqrt(
            np.sum(particle_weights * (dxs**2 + dys**2))
        )

        return weighted_std_x, weighted_std_y, weighted_std_theta, weighted_pos_std


    def evaluate_step(
        self,
        step_idx: int,
        t: float,

        true_pose: Pose2D,
        raw_odom_pose: Pose2D,

        scan_match_failed: Optional[bool],
        scan_match_fallback_failed: Optional[bool],

        particle_poses: List[Pose2D],
        particle_weights: List[float],

        particle_poses_before_resampling: List[Pose2D],
        particle_weights_before_resampling: List[float],
        neff: Optional[float],

        particle_inherit_indices: Optional[List[int]] = None,

        step_duration: Optional[float] = None,
        proposal_metrics: Optional[dict] = None,
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
        # Convert input data to numpy arrays
        true_pose = self._pose_to_np_array(true_pose)
        raw_odom_pose = self._pose_to_np_array(raw_odom_pose)
        particle_poses = self._poses_to_np_array(particle_poses)
        particle_weights = np.array(particle_weights, dtype=np.float64)
        particle_poses_before_resampling = self._poses_to_np_array(particle_poses_before_resampling)
        particle_weights_before_resampling = np.array(particle_weights_before_resampling, dtype=np.float64)
        particle_inherit_indices = (
            np.array(particle_inherit_indices, dtype=np.int32) if particle_inherit_indices is not None else None
        )

        # Validate input data
        if particle_poses is None or particle_poses.size == 0:
            raise ValueError(f"[Evaluator mp] Particle poses are None or empty at step {step_idx}.")
        if particle_weights is None or particle_weights.size == 0:
            raise ValueError(f"[Evaluator mp] Particle weights are None or empty at step {step_idx}.")
        if particle_poses_before_resampling is None or particle_poses_before_resampling.size == 0:
            raise ValueError(f"[Evaluator mp] Particle poses before resampling are None or empty at step {step_idx}.")
        if particle_weights_before_resampling is None or particle_weights_before_resampling.size == 0:
            raise ValueError(f"[Evaluator mp] Particle weights before resampling are None or empty at step {step_idx}.")
        if not np.all(np.isfinite(true_pose)):
            raise ValueError(f"[Evaluator mp] True pose contains non-finite values at step {step_idx}: {true_pose}")
        if not np.all(np.isfinite(raw_odom_pose)):
            raise ValueError(f"[Evaluator mp] Raw odometry pose contains non-finite values at step {step_idx}: {raw_odom_pose}")
        if not np.all(np.isfinite(particle_poses)):
            raise ValueError(f"[Evaluator mp] Particle poses contain non-finite values at step {step_idx}: {particle_poses}")
        if not np.all(np.isfinite(particle_weights)):
            raise ValueError(f"[Evaluator mp] Particle weights contain non-finite values at step {step_idx}: {particle_weights}")
        if not np.all(np.isfinite(particle_poses_before_resampling)):
            raise ValueError(f"[Evaluator mp] Particle poses before resampling contain non-finite values at step {step_idx}: {particle_poses_before_resampling}")
        if not np.all(np.isfinite(particle_weights_before_resampling)):
            raise ValueError(f"[Evaluator mp] Particle weights before resampling contain non-finite values at step {step_idx}: {particle_weights_before_resampling}")

        # Init step results with None values
        step_res = StepResult()

        # Add general step information
        step_res.step_idx = step_idx
        step_res.t = t
        step_res.t_step_duration = float(step_duration) if step_duration is not None else None

        # Store input arrays directly in result
        step_res.true_pose = true_pose
        step_res.raw_odom_pose = raw_odom_pose
        step_res.particle_poses = particle_poses
        step_res.particle_weights = particle_weights
        step_res.particle_poses_before_resampling = particle_poses_before_resampling
        step_res.particle_weights_before_resampling = particle_weights_before_resampling

        # Add scan match information
        step_res.scan_match_failed = scan_match_failed
        step_res.scan_match_failed_fallback = scan_match_fallback_failed

        # Store neff
        step_res.neff = float(neff)
        n_particles = len(particle_poses)
        step_res.neff_ratio = float(neff) / n_particles if n_particles > 0 else None
        
        # Store indices of particles if resampling took place, otherwise None
        step_res.particle_inherit_indices = particle_inherit_indices

        # Estimate if resampling took place
        step_res.resampling = particle_inherit_indices is not None
        step_res.unique_parents = (
            int(len(np.unique(step_res.particle_inherit_indices)))
            if step_res.particle_inherit_indices is not None
            else None
        )

        # Extract proposal metrics and underlying scan matcher and icp metrics
        if proposal_metrics is not None:
            prop_timings = proposal_metrics.get("prop_timings")
            # Extract and filter proposal timings
            if isinstance(prop_timings, dict):
                step_res.t_sample_poses = prop_timings.get("t_sample_poses")
                step_res.t_pred_poses = prop_timings.get("t_pred_poses")
                step_res.t_motion_model = prop_timings.get("t_motion_model")
                step_res.t_meas_model = prop_timings.get("t_meas_model")
                step_res.t_compute_prop_params = prop_timings.get("t_compute_prop_params")
                step_res.t_sample_from_prop = prop_timings.get("t_sample_from_prop")

            # Extract scan matcher timings
            scan_matcher_info = proposal_metrics.get("scan_matcher_info")
            if isinstance(scan_matcher_info, dict):
                # Extract scan matcher timings
                step_res.time_duration_scan_matching = scan_matcher_info.get("time_duration_scan_matching")
                step_res.time_duration_prediction = scan_matcher_info.get("time_duration_prediction")    
                step_res.time_duration_map_extraction = scan_matcher_info.get("time_duration_map_extraction")
                step_res.time_duration_correct_pose = scan_matcher_info.get("time_duration_correct_pose")
                step_res.time_duration_update_pose = scan_matcher_info.get("time_duration_update_pose")

                # Extract icp timings
                step_res.t_init_icp_trans = scan_matcher_info.get("t_init_icp_trans")
                step_res.t_init_and_train_nn_tree_normals = scan_matcher_info.get("t_init_and_train_nn_tree_normals")
                step_res.t_downsampling_pointcloud = scan_matcher_info.get("t_downsampling_pointcloud")
                step_res.t_compute_normal = scan_matcher_info.get("t_compute_normal")
                step_res.t_outlier_rejection = scan_matcher_info.get("t_outlier_rejection")
                step_res.t_find_nn_outlier_rejec = scan_matcher_info.get("t_find_nn_outlier_rejec")
                step_res.t_prepare_system = scan_matcher_info.get("t_prepare_system")
                step_res.t_solve_least_squares = scan_matcher_info.get("t_solve_least_squares")
                step_res.t_transf_update_and_results = scan_matcher_info.get("t_transf_update_and_results")
                step_res.t_find_trans = scan_matcher_info.get("t_find_trans")

        # Convert inherit indices to np array if provided
        # if particle_inherit_indices is not None:
        #     raise ValueError(f"[Evaluator mp] Particle inherit indices should be None at step {step_idx}, but got: {particle_inherit_indices}")

        # Normalize particle weights 
        # Check for NaN values in particle weights
        if np.isnan(step_res.particle_weights).any():
            raise ValueError(f"[Evaluator mp] Particle weights contain NaN values at step {step_idx}: {step_res.particle_weights}")

        # Normalize weights
        weight_sum = np.sum(step_res.particle_weights)
        
        if weight_sum <= 0:
            raise ValueError(f"[Evaluator mp] Sum of particle weights is zero or negative at step {step_idx}, cannot compute weighted mean.")
        
        step_res.particle_weights = self.norm_weights(step_res.particle_weights)

        # Compute raw odom errors
        step_res.trans_err_raw_odom = self.translation_error(step_res.raw_odom_pose, step_res.true_pose)
        step_res.rot_err_raw_odom = np.abs(self.angle_diff(step_res.raw_odom_pose[2], step_res.true_pose[2]))

        # Compute weighted mean poses
        step_res.weighted_mean_pose = self.weighted_mean_position(step_res.particle_poses_before_resampling, step_res.particle_weights_before_resampling)
        # Compute trans and rot error of weighted mean pose
        step_res.trans_err_weighted_mean = self.translation_error(step_res.weighted_mean_pose, step_res.true_pose)
        step_res.rot_err_weighted_mean = np.abs(self.angle_diff(step_res.weighted_mean_pose[2], step_res.true_pose[2]))

        # Compute weighted standard deviation of particle poses
        (
            step_res.weighted_part_std_x,
            step_res.weighted_part_std_y,
            step_res.weighted_part_std_theta,
            step_res.weighted_part_std_pos,
        ) = self.compute_weighted_part_std(
            particle_poses=step_res.particle_poses_before_resampling,
            particle_weights=step_res.particle_weights_before_resampling,
            weighted_mean_poses=step_res.weighted_mean_pose
        )

        # Estimate particle with highest weight -> Will give us online trajectory in summary
        step_res.max_weight_idx = np.argmax(step_res.particle_weights_before_resampling)
        step_res.best_particle_pose = step_res.particle_poses_before_resampling[step_res.max_weight_idx]
        step_res.best_particle_weight = step_res.particle_weights_before_resampling[step_res.max_weight_idx]
        # Compute trans and rot error of particle with highest weight
        step_res.trans_err_best_particle = self.translation_error(step_res.best_particle_pose, step_res.true_pose)
        step_res.rot_err_best_particle = np.abs(self.angle_diff(step_res.best_particle_pose[2], step_res.true_pose[2]))

        # Estimate closest particle before resampling -> Will give us offline trajectory in summary
        xy_diffs_before_resampling = step_res.particle_poses_before_resampling[:, :2] - step_res.true_pose[:2]
        trans_errors_before_resampling = np.linalg.norm(xy_diffs_before_resampling, axis=1)

        rot_diffs_before_resampling = step_res.particle_poses_before_resampling[:, 2] - step_res.true_pose[2]
        rot_errs_before_resampling = np.abs(
            np.arctan2(
                np.sin(rot_diffs_before_resampling),
                np.cos(rot_diffs_before_resampling),
            )
        )

        # Find closest particle before resampling
        step_res.idx_closest_p_before_resampling = np.argmin(trans_errors_before_resampling)
        step_res.trans_err_closest_p_before_resampling = trans_errors_before_resampling[step_res.idx_closest_p_before_resampling]
        step_res.rot_err_closest_p_before_resampling = rot_errs_before_resampling[step_res.idx_closest_p_before_resampling]
        step_res.gap_trans_best_p_to_closest_before_resamp = (
            step_res.trans_err_best_particle - step_res.trans_err_closest_p_before_resampling
        )
        step_res.gap_rot_best_p_to_closest_before_resamp = (
            step_res.rot_err_best_particle - step_res.rot_err_closest_p_before_resampling
        )

        # Estimate closest particle after resampling
        xy_diffs_after_resampling = step_res.particle_poses[:, :2] - step_res.true_pose[:2]
        trans_errors_after_resampling = np.linalg.norm(xy_diffs_after_resampling, axis=1)

        rot_diffs_after_resampling = step_res.particle_poses[:, 2] - step_res.true_pose[2]
        rot_errs_after_resampling = np.abs(
            np.arctan2(
                np.sin(rot_diffs_after_resampling),
                np.cos(rot_diffs_after_resampling),
            )
        )

        # Find closest particle after resampling
        step_res.idx_closest_p_after_resampling = np.argmin(trans_errors_after_resampling)
        step_res.trans_err_closest_p_after_resampling = trans_errors_after_resampling[step_res.idx_closest_p_after_resampling]
        step_res.rot_err_closest_p_after_resampling = rot_errs_after_resampling[step_res.idx_closest_p_after_resampling]
        step_res.trans_closest_p_after_before_resamp = (
            step_res.trans_err_closest_p_after_resampling - step_res.trans_err_closest_p_before_resampling
        )
        step_res.rot_closest_p_after_before_resamp = (
            step_res.rot_err_closest_p_after_resampling - step_res.rot_err_closest_p_before_resampling
        )

        # Store trans and rot errors of particle before/after resampling in step data
        step_res.trans_errs_before_resampling = trans_errors_before_resampling
        step_res.rot_errs_before_resampling = rot_errs_before_resampling
        step_res.trans_errs_after_resampling = trans_errors_after_resampling
        step_res.rot_errs_after_resampling = rot_errs_after_resampling

        # Estimate correlations
        corr_trans_weights = self._safe_spearman_correlation(
            step_res.particle_weights_before_resampling,
            -step_res.trans_errs_before_resampling,
        )

        corr_rot_weights = self._safe_spearman_correlation(
            step_res.particle_weights_before_resampling,
            -step_res.rot_errs_before_resampling,
        )

        # Transform into positive corr [-1, 1] -> [0, 2] (2 worst, 0 best)
        step_res.corr_trans_weights_pos = 1 - corr_trans_weights if corr_trans_weights is not None else None
        step_res.corr_rot_weights_pos = 1 - corr_rot_weights if corr_rot_weights is not None else None

        # Compute gap between best particle and min trans error before resampling
        step_res.gap_trans_best_to_min_before_resamp = step_res.trans_err_best_particle - np.min(step_res.trans_errs_before_resampling)
        step_res.gap_rot_best_to_min_before_resamp = step_res.rot_err_best_particle - np.min(step_res.rot_errs_before_resampling)

        # Compute improvement
        # Improvement over raw odom
        step_res.trans_err_weighted_mean_impr_over_raw_odom = self.compute_improvement(
            step_res.trans_err_weighted_mean,
            step_res.trans_err_raw_odom,
        )

        step_res.rot_err_weighted_mean_impr_over_raw_odom = self.compute_improvement(
            step_res.rot_err_weighted_mean,
            step_res.rot_err_raw_odom,
        )

        return step_res


    @staticmethod
    def _restore_particle_trajectory(
        particle_idx: int,
        particle_inherit_indices: List[Optional[np.ndarray]],
        particle_poses: List[np.ndarray],
    ) -> Tuple[np.ndarray]:
        """
        Restores the trajectory of the given particle starting at the end of the given poses list and propagating back to the 
        first poses.

        Parameters
        ----------
        particle_idx : int
            The index of the particle at the last step of the run, which trajectory should be reconstructed.
        particle_inherit_indices : List[Optional[np.ndarray]]
            A list of arrays containing the indices of the parent particles for each step. If no resampling took place at 
            a step, the corresponding entry is None.  
        particle_poses : List[np.ndarray]
            A list of arrays containing the poses of all particles in each iteration.
        
        Returns
        -------
        tuple[np.ndarray]
            The reconstructed trajectory of the given particle index.
        """
        # Init particle index with best particle index from last iteration
        p_idx = int(particle_idx)

        trajectory = []

        # Estimate the last index
        n_poses = len(particle_poses)
        last_idx = n_poses - 1

        for step_idx in reversed(range(n_poses)):
            if step_idx < last_idx:
                inherint_indices = particle_inherit_indices[step_idx]

                if inherint_indices is not None:
                    p_idx = inherint_indices[p_idx]
            
            trajectory.append(particle_poses[step_idx][p_idx])

        
        # Establish time chronological order of poses
        trajectory = np.asarray(trajectory[::-1], dtype=float)

        # Exclude NaN values
        valid_mask = np.all(np.isfinite(trajectory), axis=1) 
        trajectory = trajectory[valid_mask]

        return trajectory


    @staticmethod
    def _restore_map_trajectory_errors(
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
            A list of arrays containing the translational errors before resampling for each particle, in each step.
        rot_errs_before_resampling_list : List[np.ndarray]
            A list of arrays containing the rotational errors before resampling for each particle, in each step.
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Two numpy arrays containing the cleaned translational and rotational errors of the MAP trajectory in chronological
            order.
        """
        # Init particle index with best particle index from last iteration
        p_idx = int(best_particle_idx)

        trans_errs = []
        rot_errs = []

        # Estimate the last index
        last_idx = len(trans_errs_before_resampling_list) - 1
        
        for step_idx in reversed(range(len(trans_errs_before_resampling_list))):
            # Update inherint and particle idx
            if step_idx < last_idx:
                # Update inhering indices
                inherint_indices = particle_inherit_indices[step_idx]
            
                if inherint_indices is not None:
                    p_idx = inherint_indices[p_idx]
            
            # Update the translational and rotational errors of the MAP trajectory
            trans_errs.append(trans_errs_before_resampling_list[step_idx][p_idx])
            rot_errs.append(rot_errs_before_resampling_list[step_idx][p_idx])

        # Establish time chronological order of errors
        trans_errs = np.asarray(trans_errs[::-1], dtype=float)
        rot_errs = np.asarray(rot_errs[::-1], dtype=float)

        # Exclude NaN values
        valid_mask = np.isfinite(trans_errs) & np.isfinite(rot_errs)
        trans_errs = trans_errs[valid_mask]
        rot_errs = rot_errs[valid_mask]

        return trans_errs, rot_errs


    @staticmethod
    def compute_windowed_slopes(
        data: np.ndarray,
        x_axis_vals: Optional[np.ndarray] = None,
        window_size: int = 10,
        stride: Optional[int] = None,
    ) -> np.ndarray:
        '''
        Computes the slopes in the given values. The window size defines how many values are getting used for the computation 
        of one slope. The stride defines the step size for the next slope computation. One can add x_axis_vals, to incorporate
        time instead of step size for example.
        Slope is computed based on direct least squares solution over all window datapoints.

        Exp:
            x_axis_vals = [0, ..., 100]
            window_size = 10
            stride = 5

            Than we use 10 values for each slope computation. 

        Parameters
        ----------
        vals : np.ndarray
            The values for which the slopes should be computed.
        x_axis_vals : Optional[np.ndarray], optional
            The x-axis values for the slope computation. If None, the indices of vals are used, by default None
        window_size : int, optional
            The number of values to use for each slope computation, by default 10
        stride : Optional[int], optional
            The step size for the next slope computation. If None, the window_size is used, by default None

        Returns
        -------
        np.ndarray
            The computed slopes for each window in the given data.
        '''
        # Transform to numpy array and filter non-finite values
        data = np.asarray(data, dtype=float)

        if x_axis_vals is None:
            x_axis_vals = np.arange(data.size, dtype=float)
        else:
            x_axis_vals = np.asarray(x_axis_vals, dtype=float)

        valid = np.isfinite(data) & np.isfinite(x_axis_vals)
        data = data[valid]
        x_axis_vals = x_axis_vals[valid]

        # Check if there are enough values for at least one slope computation
        if data.size < window_size:
            return np.array([], dtype=float)

        if stride is None:
            stride = window_size

        slopes = []

        # Compute slope for each window in stride steps
        for start in range(0, data.size - window_size + 1, stride):
            end = start + window_size

            # Compute linear regression slope for the current window (direct solution)
            e_win = data[start:end]
            x_win = x_axis_vals[start:end]

            x_centered = x_win - np.mean(x_win)
            e_centered = e_win - np.mean(e_win)

            denom = np.sum(x_centered ** 2)

            if denom <= 0:
                continue

            slope = np.sum(x_centered * e_centered) / denom
            slopes.append(float(slope))

        return np.asarray(slopes, dtype=float)


    @staticmethod
    def compute_trajectory_motion_errors(
        est_poses: np.ndarray,
        true_poses: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes step-to-step motion errors between an estimated trajectory and
        the true trajectory.

        For each consecutive pair of poses:

            trans_motion_err = abs(est_delta_trans - true_delta_trans)
            rot_motion_err   = abs(est_delta_rot - true_delta_rot)

        Translation unit: m / step
        Rotation unit: rad / step

        These metrics measure trajectory smoothness / jumpiness relative to
        the true robot motion. Large values mean the estimated trajectory made
        a motion jump that does not match the real motion.
        """
        est_poses = np.asarray(est_poses, dtype=float)
        true_poses = np.asarray(true_poses, dtype=float)

        if est_poses.ndim != 2 or est_poses.shape[1] != 3:
            return np.array([], dtype=float), np.array([], dtype=float)

        if true_poses.ndim != 2 or true_poses.shape[1] != 3:
            return np.array([], dtype=float), np.array([], dtype=float)

        if est_poses.shape[0] != true_poses.shape[0]:
            return np.array([], dtype=float), np.array([], dtype=float)

        if est_poses.shape[0] < 2:
            return np.array([], dtype=float), np.array([], dtype=float)

        valid_pose_mask = (
            np.all(np.isfinite(est_poses), axis=1)
            & np.all(np.isfinite(true_poses), axis=1)
        )

        est_poses = est_poses[valid_pose_mask]
        true_poses = true_poses[valid_pose_mask]

        if est_poses.shape[0] < 2:
            return np.array([], dtype=float), np.array([], dtype=float)

        trans_motion_errors = []
        rot_motion_errors = []

        for i in range(1, est_poses.shape[0]):
            est_prev = est_poses[i - 1]
            est_curr = est_poses[i]

            true_prev = true_poses[i - 1]
            true_curr = true_poses[i]

            est_delta_trans = np.hypot(
                est_curr[X] - est_prev[X],
                est_curr[Y] - est_prev[Y],
            )
            true_delta_trans = np.hypot(
                true_curr[X] - true_prev[X],
                true_curr[Y] - true_prev[Y],
            )

            trans_motion_err = abs(est_delta_trans - true_delta_trans)

            est_delta_rot = RBPFEValMultParticles.angle_diff(
                est_curr[THETA],
                est_prev[THETA],
            )
            true_delta_rot = RBPFEValMultParticles.angle_diff(
                true_curr[THETA],
                true_prev[THETA],
            )

            rot_motion_err = abs(
                RBPFEValMultParticles.angle_diff(est_delta_rot, true_delta_rot)
            )

            trans_motion_errors.append(float(trans_motion_err))
            rot_motion_errors.append(float(rot_motion_err))

        return (
            np.asarray(trans_motion_errors, dtype=float),
            np.asarray(rot_motion_errors, dtype=float),
        )

    def summarize_run(
            self,
            step_res: List[StepResult],
            init_counter: int,
            particle_update_counter: int,
            params: Optional[ExperimentParams] = None
    ) -> Dict:
        # Summarize and filter
        # ________________________________________________________________________________________

        init_counter = int(init_counter)
        particle_update_counter = int(particle_update_counter)

        # Filter scan matcher information
        scan_match_failed_count = sum(1 for s in step_res if s.scan_match_failed)
        scan_match_fallback_failed_count = sum(1 for s in step_res if s.scan_match_failed_fallback)

        # Summarize particles
        particle_poses_before_resampling_list = [s.particle_poses_before_resampling for s in step_res]

        # Summarize particle poses and weights -> trajectory and weight lists
        # Trajectory before resampling Before resampling
        traj_p_before_resamp = self._poses_to_np_array(
            [s.particle_poses_before_resampling for s in step_res]
        )
        traj_w_before_resamp = self._poses_to_np_array(
            [s.particle_weights_before_resampling for s in step_res]
        )
        
        # Trajectory true poses
        traj_true_poses = self._poses_to_np_array(
            [s.true_pose for s in step_res]
        )
        
        # Trajectory weighted mean poses
        traj_weighted_mean_poses = self._poses_to_np_array(
            [s.weighted_mean_pose for s in step_res]
        )   

        # Trajectory best particle poses
        traj_best_particle_poses = self._poses_to_np_array(
            [s.best_particle_pose for s in step_res]
        )

        # Summarize/Filter pose errors
        # Raw odom
        trans_errs_raw_odom_unclean = [step.trans_err_raw_odom for step in step_res]
        rot_errs_raw_odom_unclean = [step.rot_err_raw_odom for step in step_res]
        trans_errs_raw_odom = self._finite_values(trans_errs_raw_odom_unclean)
        rot_errs_raw_odom = self._finite_values(rot_errs_raw_odom_unclean)

        # Summarize/Filter weighted mean pose errors
        trans_errs_weighted_mean = self._finite_values([step.trans_err_weighted_mean for step in step_res])
        rot_errs_weighted_mean = self._finite_values([step.rot_err_weighted_mean for step in step_res])
        weighted_part_std_theta_values = self._finite_values(
            [step.weighted_part_std_theta for step in step_res if step.weighted_part_std_theta is not None]
        )
        weighted_part_std_pos_values = self._finite_values(
            [step.weighted_part_std_pos for step in step_res if step.weighted_part_std_pos is not None]
        )

        # Summarize/Filter Best particle pose trajectory metrics
        trans_errs_best_particle = self._finite_values([step.trans_err_best_particle for step in step_res])
        rot_errs_best_particle = self._finite_values([step.rot_err_best_particle for step in step_res])
        best_particle_weight_values = self._finite_values(
            [step.best_particle_weight for step in step_res if step.best_particle_weight is not None]
        )

        # Summarize/Filter Closest particle pose trajectory metrics
        trans_errs_closest_p_before_resampling = self._finite_values(
            [step.trans_err_closest_p_before_resampling for step in step_res]
        )
        rot_errs_closest_p_before_resampling = self._finite_values(
            [step.rot_err_closest_p_before_resampling for step in step_res]
        )
        gap_trans_best_p_to_closest_before_resamp_values = self._finite_values(
            [step.gap_trans_best_p_to_closest_before_resamp for step in step_res]
        )
        gap_rot_best_p_to_closest_before_resamp_values = self._finite_values(
            [step.gap_rot_best_p_to_closest_before_resamp for step in step_res]
        )
        gap_rot_best_p_to_closest_before_resamp_values = np.abs(
            gap_rot_best_p_to_closest_before_resamp_values
        )
        gap_trans_best_to_min_before_resamp_values = self._finite_values(
            [step.gap_trans_best_to_min_before_resamp for step in step_res]
        )
        gap_rot_best_to_min_before_resamp_values = self._finite_values(
            [step.gap_rot_best_to_min_before_resamp for step in step_res]
        )
        trans_errs_closest_p_after_resampling = self._finite_values(
            [step.trans_err_closest_p_after_resampling for step in step_res]
        )
        rot_errs_closest_p_after_resampling = self._finite_values(
            [step.rot_err_closest_p_after_resampling for step in step_res]
        )

        # Summarize/Filter Closest particle pose trajectory metrics
        trans_closest_p_after_before_resamp_vals = self._finite_values(
            [
                step.trans_closest_p_after_before_resamp
                for step in step_res
                if step.resampling and step.trans_closest_p_after_before_resamp is not None
            ]
        )
        rot_closest_p_after_before_resamp_vals = self._finite_values(
            [
                step.rot_closest_p_after_before_resamp
                for step in step_res
                if step.resampling and step.rot_closest_p_after_before_resamp is not None
            ]
        )
        rot_closest_p_after_before_resamp_vals = np.abs(
            rot_closest_p_after_before_resamp_vals
        )
        corr_trans_weights_pos_values = self._finite_values(
            [step.corr_trans_weights_pos for step in step_res]
        )
        corr_rot_weights_pos_values = self._finite_values(
            [step.corr_rot_weights_pos for step in step_res]
        )

        # Filter improvement metrics
        trans_errs_weighted_mean_impr_over_raw_odom = self._finite_values(
            [step.trans_err_weighted_mean_impr_over_raw_odom for step in step_res]
        )
        rot_errs_weighted_mean_impr_over_raw_odom = self._finite_values(
            [step.rot_err_weighted_mean_impr_over_raw_odom for step in step_res]
        )
        # rot_errs_weighted_mean_impr_over_raw_odom = np.abs(
        #     rot_errs_weighted_mean_impr_over_raw_odom
        # )

        # Filter time durations
        step_durations = self._finite_values([s.t_step_duration for s in step_res if s.t_step_duration is not None])
        # Filter and store proposal time durations
        t_sample_poses_values = self._finite_values(
            [s.t_sample_poses for s in step_res if s.t_sample_poses is not None]
        )
        t_pred_poses_values = self._finite_values(
            [s.t_pred_poses for s in step_res if s.t_pred_poses is not None]
        )
        t_motion_model_values = self._finite_values(
            [s.t_motion_model for s in step_res if s.t_motion_model is not None]
        )
        t_meas_model_values = self._finite_values(
            [s.t_meas_model for s in step_res if s.t_meas_model is not None]
        )
        t_compute_prop_params_values = self._finite_values(
            [s.t_compute_prop_params for s in step_res if s.t_compute_prop_params is not None]
        )
        t_sample_from_prop_values = self._finite_values(
            [s.t_sample_from_prop for s in step_res if s.t_sample_from_prop is not None]
        )
        
        # Filter and store time durations scan matching
        time_duration_scan_matching_values = self._finite_values(
            [s.time_duration_scan_matching for s in step_res if s.time_duration_scan_matching is not None]
        )
        time_duration_prediction_values = self._finite_values(
            [s.time_duration_prediction for s in step_res if s.time_duration_prediction is not None]
        )
        time_duration_map_extraction_values = self._finite_values(
            [s.time_duration_map_extraction for s in step_res if s.time_duration_map_extraction is not None]
        )
        time_duration_correct_pose_values = self._finite_values(
            [s.time_duration_correct_pose for s in step_res if s.time_duration_correct_pose is not None]
        )
        time_duration_update_pose_values = self._finite_values(
            [s.time_duration_update_pose for s in step_res if s.time_duration_update_pose is not None]
        )

        # Filter and store time durations ICP
        t_init_icp_trans_values = self._finite_values(
            [s.t_init_icp_trans for s in step_res if s.t_init_icp_trans is not None]
        )
        t_init_and_train_nn_tree_normals_values = self._finite_values(
            [s.t_init_and_train_nn_tree_normals for s in step_res if s.t_init_and_train_nn_tree_normals is not None]
        )
        t_downsampling_pointcloud_values = self._finite_values(
            [s.t_downsampling_pointcloud for s in step_res if s.t_downsampling_pointcloud is not None]
        )
        t_compute_normal_values = self._finite_values(
            [s.t_compute_normal for s in step_res if s.t_compute_normal is not None]
        )
        t_outlier_rejection_values = self._finite_values(
            [s.t_outlier_rejection for s in step_res if s.t_outlier_rejection is not None]
        )
        t_find_nn_outlier_rejec_values = self._finite_values(
            [s.t_find_nn_outlier_rejec for s in step_res if s.t_find_nn_outlier_rejec is not None]
        )
        t_prepare_system_values = self._finite_values(
            [s.t_prepare_system for s in step_res if s.t_prepare_system is not None]
        )
        t_solve_least_squares_values = self._finite_values(
            [s.t_solve_least_squares for s in step_res if s.t_solve_least_squares is not None]
        )
        t_transf_update_and_results_values = self._finite_values(
            [s.t_transf_update_and_results for s in step_res if s.t_transf_update_and_results is not None]
        )
        t_find_trans_values = self._finite_values(
            [s.t_find_trans for s in step_res if s.t_find_trans is not None]
        )

        # Filter neff
        neff_values = self._finite_values([step.neff for step in step_res if step.neff is not None])
        neff_ratio_values = self._finite_values([step.neff_ratio for step in step_res if step.neff_ratio is not None])
        unique_resampled_parents_values = self._finite_values(
            [step.unique_parents for step in step_res if step.unique_parents is not None]
        )

        # Summarize/filter trans errors before resampling
        trans_errs_before_resampling_list = [step.trans_errs_before_resampling for step in step_res]
        rot_errs_before_resampling_list = [step.rot_errs_before_resampling for step in step_res]

        # Restore MAP trajectory errors
        # ________________________________________________________________________________________

        best_p_idx = step_res[-1].max_weight_idx
        particle_inherit_indices = [step.particle_inherit_indices for step in step_res]

        traj_particle_map = self._restore_particle_trajectory(
            particle_idx=best_p_idx,
            particle_inherit_indices=particle_inherit_indices,
            particle_poses=particle_poses_before_resampling_list,
        )

        trans_errs_map_traj = self.trans_err_trajectory(traj_particle_map, traj_true_poses)
        rot_errs_map_traj = self.rot_err_trajectory(traj_particle_map, traj_true_poses)

        
        # Compute final map improvement
        # ________________________________________________________________________________________
        
        # Validate that the compared error metrics have same len
        if len(trans_errs_map_traj) != len(trans_errs_raw_odom_unclean):
            raise ValueError(
                f"[Evaluator mp] Mismatch in number of steps between map trajectory errors and raw odometry errors. "
                f"Map trajectory errors: {len(trans_errs_map_traj)}, Raw odometry errors: {len(trans_errs_raw_odom_unclean)}"
            )

        if len(rot_errs_map_traj) != len(rot_errs_raw_odom_unclean):
            raise ValueError(
                f"[Evaluator mp] Mismatch in number of steps between map trajectory errors and raw odometry errors. "
                f"Map trajectory errors: {len(rot_errs_map_traj)}, Raw odometry errors: {len(rot_errs_raw_odom_unclean)}"
            )

        trans_errs_map_traj_impr_over_raw_odom = [
            self.compute_improvement(map_err, raw_odom_err)
            for map_err, raw_odom_err in zip(trans_errs_map_traj, trans_errs_raw_odom_unclean)
        ]

        rot_errs_map_traj_impr_over_raw_odom = [
            self.compute_improvement(map_err, raw_odom_err)
            for map_err, raw_odom_err in zip(rot_errs_map_traj, rot_errs_raw_odom_unclean)
        ]

        # Ensure finite values in map traj
        trans_errs_map_traj = self._finite_values(trans_errs_map_traj)
        rot_errs_map_traj = self._finite_values(rot_errs_map_traj)

        # Clean improvement metrics
        trans_errs_map_traj_impr_over_raw_odom = self._finite_values(
            trans_errs_map_traj_impr_over_raw_odom
        )
        rot_errs_map_traj_impr_over_raw_odom = self._finite_values(
            rot_errs_map_traj_impr_over_raw_odom
        )
        # rot_errs_map_traj_impr_over_raw_odom = np.abs(
        #     rot_errs_map_traj_impr_over_raw_odom
        # )

        # Compute error slopes
        # ________________________________________________________________________________________

        # Error slopes wieghted mean
        trans_err_slopes_weighted_mean = self.compute_windowed_slopes(
            data=trans_errs_weighted_mean,
            window_size=10,
        )

        rot_err_slopes_weighted_mean = self.compute_windowed_slopes(
            data=rot_errs_weighted_mean,
            window_size=10,
        )

        # Error slopes map traj
        trans_err_slopes_map_traj = self.compute_windowed_slopes(
            data=trans_errs_map_traj,
            window_size=10,
        )

        rot_err_slopes_map_traj = self.compute_windowed_slopes(
            data=rot_errs_map_traj,
            window_size=10,
        )

        # Ensure finite slopes
        trans_err_slopes_weighted_mean = self._finite_values(trans_err_slopes_weighted_mean)
        rot_err_slopes_weighted_mean = self._finite_values(rot_err_slopes_weighted_mean)
        trans_err_slopes_map_traj = self._finite_values(trans_err_slopes_map_traj)
        rot_err_slopes_map_traj = self._finite_values(rot_err_slopes_map_traj)

        # Store only positive slopes -> Rising error is what is bad and needs to be analyzed
        trans_err_slopes_weighted_mean = np.maximum(trans_err_slopes_weighted_mean, 0.0)
        rot_err_slopes_weighted_mean = np.maximum(rot_err_slopes_weighted_mean, 0.0)
        trans_err_slopes_map_traj = np.maximum(trans_err_slopes_map_traj, 0.0)
        rot_err_slopes_map_traj = np.maximum(rot_err_slopes_map_traj, 0.0)


        # Trajectory smoothness
        # ________________________________________________________________________________________

        # Compute trajectory motion errors
        trans_motion_errs_weighted_mean, rot_motion_errs_weighted_mean = self.compute_trajectory_motion_errors(
            est_poses=traj_weighted_mean_poses,
            true_poses=traj_true_poses,
        )

        trans_motion_errs_map_traj, rot_motion_errs_map_traj = self.compute_trajectory_motion_errors(
            est_poses=traj_particle_map,
            true_poses=traj_true_poses,
        )

        trans_motion_errs_best_particle, rot_motion_errs_best_particle = self.compute_trajectory_motion_errors(
            est_poses=traj_best_particle_poses,
            true_poses=traj_true_poses,
        )

        # Clean motion errors
        trans_motion_errs_weighted_mean = self._finite_values(trans_motion_errs_weighted_mean)
        rot_motion_errs_weighted_mean = self._finite_values(rot_motion_errs_weighted_mean)  

        trans_motion_errs_map_traj = self._finite_values(trans_motion_errs_map_traj)
        rot_motion_errs_map_traj = self._finite_values(rot_motion_errs_map_traj)

        trans_motion_errs_best_particle = self._finite_values(trans_motion_errs_best_particle)
        rot_motion_errs_best_particle = self._finite_values(rot_motion_errs_best_particle)

        
        # Compute resampling count
        # ________________________________________________________________________________________

        resampling_count = sum(1 for step in step_res if step.particle_inherit_indices is not None)
        best_p_is_closest_before_resamp_count = sum(
            1
            for step in step_res
            if step.max_weight_idx is not None
            and step.idx_closest_p_before_resampling is not None
            and step.max_weight_idx == step.idx_closest_p_before_resampling
        )
                
        # Compute summary flags for available metrics
        # ________________________________________________________________________________________

        has_trans_weighted_mean_errors = len(trans_errs_weighted_mean) > 0
        has_rot_weighted_mean_errors = len(rot_errs_weighted_mean) > 0
        has_trans_weighted_mean_impr_over_raw_odom = (
            len(trans_errs_weighted_mean_impr_over_raw_odom) > 0
        )
        has_rot_weighted_mean_impr_over_raw_odom = (
            len(rot_errs_weighted_mean_impr_over_raw_odom) > 0
        )
        
        has_pose_slopes_trans_weighted_mean = (len(trans_err_slopes_weighted_mean) > 0)
        has_pose_slopes_rotation_weighted_mean = (len(rot_err_slopes_weighted_mean) > 0)

        has_weighted_part_std_theta_values = len(weighted_part_std_theta_values) > 0
        has_weighted_part_std_pos_values = len(weighted_part_std_pos_values) > 0
        has_trans_raw_odom_errors = len(trans_errs_raw_odom) > 0
        has_rot_raw_odom_errors = len(rot_errs_raw_odom) > 0
        has_trans_best_particle_errors = len(trans_errs_best_particle) > 0
        has_rot_best_particle_errors = len(rot_errs_best_particle) > 0
        has_best_particle_weight_values = len(best_particle_weight_values) > 0
        has_trans_closest_particle_before_resampling_errors = (
            len(trans_errs_closest_p_before_resampling) > 0
        )
        has_rot_closest_particle_before_resampling_errors = (
            len(rot_errs_closest_p_before_resampling) > 0
        )
        has_gap_trans_best_p_to_closest_before_resamp_values = (
            len(gap_trans_best_p_to_closest_before_resamp_values) > 0
        )
        has_gap_rot_best_p_to_closest_before_resamp_values = (
            len(gap_rot_best_p_to_closest_before_resamp_values) > 0
        )
        has_gap_trans_best_to_min_before_resamp_values = (
            len(gap_trans_best_to_min_before_resamp_values) > 0
        )
        has_gap_rot_best_to_min_before_resamp_values = (
            len(gap_rot_best_to_min_before_resamp_values) > 0
        )
        has_trans_closest_particle_after_resampling_errors = (
            len(trans_errs_closest_p_after_resampling) > 0
        )
        has_rot_closest_particle_after_resampling_errors = (
            len(rot_errs_closest_p_after_resampling) > 0
        )

        has_trans_closest_p_after_before_resamp_vals = (
            len(trans_closest_p_after_before_resamp_vals) > 0
        )
        has_rot_closest_p_after_before_resamp_vals = (
            len(rot_closest_p_after_before_resamp_vals) > 0
        )
        has_corr_trans_weights_pos_values = len(corr_trans_weights_pos_values) > 0
        has_corr_rot_weights_pos_values = len(corr_rot_weights_pos_values) > 0
        has_map_trajectory_errors = (
            len(trans_errs_map_traj) > 0 and len(rot_errs_map_traj) > 0
        )
        # has_map_trajectory_impr_over_raw_odom = (
        #     len(trans_errs_map_traj_impr_over_raw_odom) > 0
        #     and len(rot_errs_map_traj_impr_over_raw_odom) > 0
        # )

        has_trans_map_traj_impr_over_raw_odom = (
            len(trans_errs_map_traj_impr_over_raw_odom) > 0
        )

        has_rot_map_traj_impr_over_raw_odom = (
            len(rot_errs_map_traj_impr_over_raw_odom) > 0
        )


        has_pos_slope_trans_map_traj = len(trans_err_slopes_map_traj) > 0
        has_pos_slope_rot_map_traj = len(rot_err_slopes_map_traj) > 0
        has_trans_motion_errors_weighted_mean = len(trans_motion_errs_weighted_mean) > 0
        has_rot_motion_errors_weighted_mean = len(rot_motion_errs_weighted_mean) > 0
        has_trans_motion_errors_map_traj = len(trans_motion_errs_map_traj) > 0
        has_rot_motion_errors_map_traj = len(rot_motion_errs_map_traj) > 0
        has_trans_motion_errors_best_particle = len(trans_motion_errs_best_particle) > 0
        has_rot_motion_errors_best_particle = len(rot_motion_errs_best_particle) > 0
        
        has_neff_values = len(neff_values) > 0
        has_neff_ratio_values = len(neff_ratio_values) > 0
        has_unique_resampled_parents_values = len(unique_resampled_parents_values) > 0

        # Compute summary if step results are available
        # ________________________________________________________________________________________

        summary = {
            "n_steps": len(step_res),
            "init_counter": init_counter,
            "particle_update_counter": particle_update_counter,

            # Compute mean timings
            # ________________________________________________________________________________________
            "mean_step_duration": self._safe_mean(step_durations),
            # Mean time durations proposal
            "mean_t_sample_poses": self._safe_mean(t_sample_poses_values),
            "mean_t_pred_poses": self._safe_mean(t_pred_poses_values),
            "mean_t_motion_model": self._safe_mean(t_motion_model_values),
            "mean_t_meas_model": self._safe_mean(t_meas_model_values),
            "mean_t_compute_prop_params": self._safe_mean(t_compute_prop_params_values),
            "mean_t_sample_from_prop": self._safe_mean(t_sample_from_prop_values),
            # Mean time durations scan matching
            "mean_time_duration_scan_matching": self._safe_mean(time_duration_scan_matching_values),
            "mean_time_duration_prediction": self._safe_mean(time_duration_prediction_values),
            "mean_time_duration_map_extraction": self._safe_mean(time_duration_map_extraction_values),
            "mean_time_duration_correct_pose": self._safe_mean(time_duration_correct_pose_values),
            "mean_time_duration_update_pose": self._safe_mean(time_duration_update_pose_values),
            # Mean time durations ICP
            "mean_t_init_icp_trans": self._safe_mean(t_init_icp_trans_values),
            "mean_t_init_and_train_nn_tree_normals": self._safe_mean(t_init_and_train_nn_tree_normals_values),
            "mean_t_downsampling_pointcloud": self._safe_mean(t_downsampling_pointcloud_values),
            "mean_t_compute_normal": self._safe_mean(t_compute_normal_values),
            "mean_t_outlier_rejection": self._safe_mean(t_outlier_rejection_values),
            "mean_t_find_nn_outlier_rejec": self._safe_mean(t_find_nn_outlier_rejec_values),
            "mean_t_prepare_system": self._safe_mean(t_prepare_system_values),
            "mean_t_solve_least_squares": self._safe_mean(t_solve_least_squares_values),
            "mean_t_transf_update_and_results": self._safe_mean(t_transf_update_and_results_values),
            "mean_t_find_trans": self._safe_mean(t_find_trans_values),

            # Compute scan matcher statistics
            # ________________________________________________________________________________________
            "scan_match_failed_count": int(scan_match_failed_count),
            "scan_match_failed_rate": float(scan_match_failed_count) / particle_update_counter if particle_update_counter else None,
            "scan_match_fallback_failed_count": int(scan_match_fallback_failed_count),
            "scan_match_fallback_failed_rate": float(scan_match_fallback_failed_count) / particle_update_counter if particle_update_counter else None,

            # Raw odom trajectory
            # ________________________________________________________________________________________
            "mean_trans_err_raw_odom": (
                float(np.mean(trans_errs_raw_odom))
                if has_trans_raw_odom_errors else None
            ),
            "rmse_trans_err_raw_odom": (
                float(np.sqrt(np.mean(np.square(trans_errs_raw_odom))))
                if has_trans_raw_odom_errors else None
            ),
            "worst_trans_err_raw_odom": (
                float(np.max(trans_errs_raw_odom))
                if has_trans_raw_odom_errors else None
            ),
            "mean_rot_err_raw_odom": (
                float(np.mean(rot_errs_raw_odom))
                if has_rot_raw_odom_errors else None
            ),
            "rmse_rot_err_raw_odom": (
                float(np.sqrt(np.mean(np.square(rot_errs_raw_odom))))
                if has_rot_raw_odom_errors else None
            ),
            "worst_rot_err_raw_odom": (
                float(np.max(rot_errs_raw_odom))
                if has_rot_raw_odom_errors else None
            ),
            "final_trans_drift_trans_err_raw_odom": (
                float(trans_errs_raw_odom[-1])
                if has_trans_raw_odom_errors else None
            ),
            "final_rot_drift_rot_err_raw_odom": (
                float(rot_errs_raw_odom[-1])
                if has_rot_raw_odom_errors else None
            ),

            # Trajectory before resampling
            # ________________________________________________________________________________________
            # Correlations
            "mean_corr_trans_weights_pos": (
                float(np.mean(corr_trans_weights_pos_values))
                if has_corr_trans_weights_pos_values else None
            ),
            "median_corr_trans_weights_pos": (
                float(np.median(corr_trans_weights_pos_values))
                if has_corr_trans_weights_pos_values else None
            ),
            "mean_corr_rot_weights_pos": (
                float(np.mean(corr_rot_weights_pos_values))
                if has_corr_rot_weights_pos_values else None
            ),
            "median_corr_rot_weights_pos": (
                float(np.median(corr_rot_weights_pos_values))
                if has_corr_rot_weights_pos_values else None
            ),

            # Weighted mean pose trajectory
            # ________________________________________________________________________________________
            "mean_trans_err_weighted_mean": (
                float(np.mean(trans_errs_weighted_mean))
                if has_trans_weighted_mean_errors else None
            ),
            "rmse_trans_err_weighted_mean": (
                float(np.sqrt(np.mean(np.square(trans_errs_weighted_mean))))
                if has_trans_weighted_mean_errors else None
            ),
            "worst_trans_err_weighted_mean": (
                float(np.max(trans_errs_weighted_mean))
                if has_trans_weighted_mean_errors else None
            ),
            "mean_rot_err_weighted_mean": (
                float(np.mean(rot_errs_weighted_mean))
                if has_rot_weighted_mean_errors else None
            ),
            "rmse_rot_err_weighted_mean": (
                float(np.sqrt(np.mean(np.square(rot_errs_weighted_mean))))
                if has_rot_weighted_mean_errors else None
            ),
            "worst_rot_err_weighted_mean": (
                float(np.max(rot_errs_weighted_mean))
                if has_rot_weighted_mean_errors else None
            ),
            "final_trans_drift_trans_err_weighted_mean": (
                float(trans_errs_weighted_mean[-1])
                if has_trans_weighted_mean_errors else None
            ),
            "final_rot_drift_rot_err_weighted_mean": (
                float(rot_errs_weighted_mean[-1])
                if has_rot_weighted_mean_errors else None
            ),
            "mean_trans_motion_err_weighted_mean": (
                float(np.mean(trans_motion_errs_weighted_mean))
                if has_trans_motion_errors_weighted_mean else None
            ),
            "rmse_trans_motion_err_weighted_mean": (
                float(np.sqrt(np.mean(np.square(trans_motion_errs_weighted_mean))))
                if has_trans_motion_errors_weighted_mean else None
            ),
            "worst_trans_motion_err_weighted_mean": (
                float(np.max(trans_motion_errs_weighted_mean))
                if has_trans_motion_errors_weighted_mean else None
            ),
            "p90_trans_motion_err_weighted_mean": (
                float(np.percentile(trans_motion_errs_weighted_mean, 90))
                if has_trans_motion_errors_weighted_mean else None
            ),
            "mean_rot_motion_err_weighted_mean": (
                float(np.mean(rot_motion_errs_weighted_mean))
                if has_rot_motion_errors_weighted_mean else None
            ),
            "rmse_rot_motion_err_weighted_mean": (
                float(np.sqrt(np.mean(np.square(rot_motion_errs_weighted_mean))))
                if has_rot_motion_errors_weighted_mean else None
            ),
            "worst_rot_motion_err_weighted_mean": (
                float(np.max(rot_motion_errs_weighted_mean))
                if has_rot_motion_errors_weighted_mean else None
            ),
            "p90_rot_motion_err_weighted_mean": (
                float(np.percentile(rot_motion_errs_weighted_mean, 90))
                if has_rot_motion_errors_weighted_mean else None
            ),
            # improvement of weighted mean over raw odom
            "mean_trans_err_weighted_mean_impr_over_raw_odom": (
                float(np.mean(trans_errs_weighted_mean_impr_over_raw_odom))
                if has_trans_weighted_mean_impr_over_raw_odom else None
            ),
            "median_trans_err_weighted_mean_impr_over_raw_odom": (
                float(np.median(trans_errs_weighted_mean_impr_over_raw_odom))
                if has_trans_weighted_mean_impr_over_raw_odom else None
            ),
            "perc_10_trans_err_weighted_mean_impr_over_raw_odom": (
                float(np.percentile(trans_errs_weighted_mean_impr_over_raw_odom, 10))
                if has_trans_weighted_mean_impr_over_raw_odom else None
            ),
            # Improvement of weighted mean over raw odom
            "mean_rot_err_weighted_mean_impr_over_raw_odom": (
                float(np.mean(rot_errs_weighted_mean_impr_over_raw_odom))
                if has_rot_weighted_mean_impr_over_raw_odom else None
            ),
            "median_rot_err_weighted_mean_impr_over_raw_odom": (
                float(np.median(rot_errs_weighted_mean_impr_over_raw_odom))
                if has_rot_weighted_mean_impr_over_raw_odom else None
            ),
            "perc_10_rot_err_weighted_mean_impr_over_raw_odom": (
                float(np.percentile(rot_errs_weighted_mean_impr_over_raw_odom, 10))
                if has_rot_weighted_mean_impr_over_raw_odom else None
            ),
            # Compute slope for weighted mean errors
            "mean_pos_trans_errs_slopes_weighted_mean": (
                float(np.mean(trans_err_slopes_weighted_mean))
                if has_pose_slopes_trans_weighted_mean else None
            ),
            "rmse_pos_trans_errs_slopes_weighted_mean": (
                float(np.sqrt(np.mean(np.square(trans_err_slopes_weighted_mean))))
                if has_pose_slopes_trans_weighted_mean else None
            ),
            "p90_pos_trans_errs_slopes_weighted_mean": (
                float(np.percentile(trans_err_slopes_weighted_mean, 90))
                if has_pose_slopes_trans_weighted_mean else None
            ),
            "mean_pos_rot_errs_slopes_weighted_mean": (
                float(np.mean(rot_err_slopes_weighted_mean))
                if has_pose_slopes_rotation_weighted_mean else None
            ),
            "rmse_pos_rot_errs_slopes_weighted_mean": (
                float(np.sqrt(np.mean(np.square(rot_err_slopes_weighted_mean))))
                if has_pose_slopes_rotation_weighted_mean else None
            ),
            "p90_pos_rot_errs_slopes_weighted_mean": (
                float(np.percentile(rot_err_slopes_weighted_mean, 90))
                if has_pose_slopes_rotation_weighted_mean else None
            ),
            "mean_weighted_part_std_theta": (
                float(np.mean(weighted_part_std_theta_values))
                if has_weighted_part_std_theta_values else None
            ),
            "max_weighted_part_std_theta": (
                float(np.max(weighted_part_std_theta_values))
                if has_weighted_part_std_theta_values else None
            ),
            "mean_weighted_part_std_pos": (
                float(np.mean(weighted_part_std_pos_values))
                if has_weighted_part_std_pos_values else None
            ),
            "max_weighted_part_std_pos": (
                float(np.max(weighted_part_std_pos_values))
                if has_weighted_part_std_pos_values else None
            ),

            # Online MAP / highest-weight particle trajectory
            # ________________________________________________________________________________________
            "mean_trans_err_best_particle": (
                float(np.mean(trans_errs_best_particle))
                if has_trans_best_particle_errors else None
            ),
            "rmse_trans_err_best_particle": (
                float(np.sqrt(np.mean(np.square(trans_errs_best_particle))))
                if has_trans_best_particle_errors else None
            ),
            "worst_trans_err_best_particle": (
                float(np.max(trans_errs_best_particle))
                if has_trans_best_particle_errors else None
            ),
            "mean_rot_err_best_particle": (
                float(np.mean(rot_errs_best_particle))
                if has_rot_best_particle_errors else None
            ),
            "rmse_rot_err_best_particle": (
                float(np.sqrt(np.mean(np.square(rot_errs_best_particle))))
                if has_rot_best_particle_errors else None
            ),
            "worst_rot_err_best_particle": (
                float(np.max(rot_errs_best_particle))
                if has_rot_best_particle_errors else None
            ),
            "final_trans_drift_trans_err_best_particle": (
                float(trans_errs_best_particle[-1])
                if has_trans_best_particle_errors else None
            ),
            "final_rot_drift_rot_err_best_particle": (
                float(rot_errs_best_particle[-1])
                if has_rot_best_particle_errors else None
            ),
            "mean_trans_motion_err_best_particle": (
                float(np.mean(trans_motion_errs_best_particle))
                if has_trans_motion_errors_best_particle else None
            ),
            "rmse_trans_motion_err_best_particle": (
                float(np.sqrt(np.mean(np.square(trans_motion_errs_best_particle))))
                if has_trans_motion_errors_best_particle else None
            ),
            "worst_trans_motion_err_best_particle": (
                float(np.max(trans_motion_errs_best_particle))
                if has_trans_motion_errors_best_particle else None
            ),
            "p90_trans_motion_err_best_particle": (
                float(np.percentile(trans_motion_errs_best_particle, 90))
                if has_trans_motion_errors_best_particle else None
            ),
            "mean_rot_motion_err_best_particle": (
                float(np.mean(rot_motion_errs_best_particle))
                if has_rot_motion_errors_best_particle else None
            ),
            "rmse_rot_motion_err_best_particle": (
                float(np.sqrt(np.mean(np.square(rot_motion_errs_best_particle))))
                if has_rot_motion_errors_best_particle else None
            ),
            "worst_rot_motion_err_best_particle": (
                float(np.max(rot_motion_errs_best_particle))
                if has_rot_motion_errors_best_particle else None
            ),
            "p90_rot_motion_err_best_particle": (
                float(np.percentile(rot_motion_errs_best_particle, 90))
                if has_rot_motion_errors_best_particle else None
            ),
            "rate_above_thres_trans_err_best_particle": (
                float(np.mean(trans_errs_best_particle > TRANS_ERRS_BEST_PARTICLE_THRES))
                if has_trans_best_particle_errors else None
            ),
            "rate_above_thres_rot_err_best_particle": (
                float(np.mean(rot_errs_best_particle > ROT_ERRS_BEST_PARTICLE_THRES))
                if has_rot_best_particle_errors else None
            ),
            "mean_best_particle_weight": (
                float(np.mean(best_particle_weight_values))
                if has_best_particle_weight_values else None
            ),
            "max_best_particle_weight": (
                float(np.max(best_particle_weight_values))
                if has_best_particle_weight_values else None
            ),
            "min_best_particle_weight": (
                float(np.min(best_particle_weight_values))
                if has_best_particle_weight_values else None
            ),
            "final_best_particle_weight": (
                float(best_particle_weight_values[-1])
                if has_best_particle_weight_values else None
            ),

            # Oracle closest-particle trajectory before resampling
            # ________________________________________________________________________________________
            "mean_trans_err_closest_p_before_resampling": (
                float(np.mean(trans_errs_closest_p_before_resampling))
                if has_trans_closest_particle_before_resampling_errors else None
            ),
            "rmse_trans_err_closest_p_before_resampling": (
                float(np.sqrt(np.mean(np.square(trans_errs_closest_p_before_resampling))))
                if has_trans_closest_particle_before_resampling_errors else None
            ),
            "worst_trans_err_closest_p_before_resampling": (
                float(np.max(trans_errs_closest_p_before_resampling))
                if has_trans_closest_particle_before_resampling_errors else None
            ),
            "mean_rot_err_closest_p_before_resampling": (
                float(np.mean(rot_errs_closest_p_before_resampling))
                if has_rot_closest_particle_before_resampling_errors else None
            ),
            "rmse_rot_err_closest_p_before_resampling": (
                float(np.sqrt(np.mean(np.square(rot_errs_closest_p_before_resampling))))
                if has_rot_closest_particle_before_resampling_errors else None
            ),
            "worst_rot_err_closest_p_before_resampling": (
                float(np.max(rot_errs_closest_p_before_resampling))
                if has_rot_closest_particle_before_resampling_errors else None
            ),
            "final_trans_drift_trans_err_closest_p_before_resampling": (
                float(trans_errs_closest_p_before_resampling[-1])
                if has_trans_closest_particle_before_resampling_errors else None
            ),
            "final_rot_drift_rot_err_closest_p_before_resampling": (
                float(rot_errs_closest_p_before_resampling[-1])
                if has_rot_closest_particle_before_resampling_errors else None
            ),
            "mean_gap_trans_best_p_to_closest_before_resamp": (
                float(np.mean(gap_trans_best_p_to_closest_before_resamp_values))
                if has_gap_trans_best_p_to_closest_before_resamp_values else None
            ),
            "rmse_gap_trans_best_p_to_closest_before_resamp": (
                float(np.sqrt(np.mean(np.square(gap_trans_best_p_to_closest_before_resamp_values))))
                if has_gap_trans_best_p_to_closest_before_resamp_values else None
            ),
            "worst_gap_trans_best_p_to_closest_before_resamp": (
                float(np.max(gap_trans_best_p_to_closest_before_resamp_values))
                if has_gap_trans_best_p_to_closest_before_resamp_values else None
            ),
            "mean_gap_rot_best_p_to_closest_before_resamp": (
                float(np.mean(gap_rot_best_p_to_closest_before_resamp_values))
                if has_gap_rot_best_p_to_closest_before_resamp_values else None
            ),
            "rmse_gap_rot_best_p_to_closest_before_resamp": (
                float(np.sqrt(np.mean(np.square(gap_rot_best_p_to_closest_before_resamp_values))))
                if has_gap_rot_best_p_to_closest_before_resamp_values else None
            ),
            "worst_gap_rot_best_p_to_closest_before_resamp": (
                float(np.max(gap_rot_best_p_to_closest_before_resamp_values))
                if has_gap_rot_best_p_to_closest_before_resamp_values else None
            ),
            "rate_gap_trans_best_to_min_before_resamp_above_max_trans_gap": (
                float(np.mean(gap_trans_best_to_min_before_resamp_values > MAX_TRANS_GAP))
                if has_gap_trans_best_to_min_before_resamp_values else None
            ),
            "rate_gap_rot_best_to_min_before_resamp_above_max_rot_gap": (
                float(np.mean(gap_rot_best_to_min_before_resamp_values > MAX_ROT_GAP))
                if has_gap_rot_best_to_min_before_resamp_values else None
            ),

            # Oracle closest-particle trajectory after resampling
            # ________________________________________________________________________________________
            "mean_trans_err_closest_p_after_resampling": (
                float(np.mean(trans_errs_closest_p_after_resampling))
                if has_trans_closest_particle_after_resampling_errors else None
            ),
            "rmse_trans_err_closest_p_after_resampling": (
                float(np.sqrt(np.mean(np.square(trans_errs_closest_p_after_resampling))))
                if has_trans_closest_particle_after_resampling_errors else None
            ),
            "worst_trans_err_closest_p_after_resampling": (
                float(np.max(trans_errs_closest_p_after_resampling))
                if has_trans_closest_particle_after_resampling_errors else None
            ),
            "mean_rot_err_closest_p_after_resampling": (
                float(np.mean(rot_errs_closest_p_after_resampling))
                if has_rot_closest_particle_after_resampling_errors else None
            ),
            "rmse_rot_err_closest_p_after_resampling": (
                float(np.sqrt(np.mean(np.square(rot_errs_closest_p_after_resampling))))
                if has_rot_closest_particle_after_resampling_errors else None
            ),
            "worst_rot_err_closest_p_after_resampling": (
                float(np.max(rot_errs_closest_p_after_resampling))
                if has_rot_closest_particle_after_resampling_errors else None
            ),
            "final_trans_drift_trans_err_closest_p_after_resampling": (
                float(trans_errs_closest_p_after_resampling[-1])
                if has_trans_closest_particle_after_resampling_errors else None
            ),
            "final_rot_drift_rot_err_closest_p_after_resampling": (
                float(rot_errs_closest_p_after_resampling[-1])
                if has_rot_closest_particle_after_resampling_errors else None
            ),


            # Final MAP particle ancestral trajectory
            # ________________________________________________________________________________________
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
            "mean_trans_motion_err_map_traj": (
                float(np.mean(trans_motion_errs_map_traj))
                if has_trans_motion_errors_map_traj else None
            ),
            "rmse_trans_motion_err_map_traj": (
                float(np.sqrt(np.mean(np.square(trans_motion_errs_map_traj))))
                if has_trans_motion_errors_map_traj else None
            ),
            "worst_trans_motion_err_map_traj": (
                float(np.max(trans_motion_errs_map_traj))
                if has_trans_motion_errors_map_traj else None
            ),
            "p90_trans_motion_err_map_traj": (
                float(np.percentile(trans_motion_errs_map_traj, 90))
                if has_trans_motion_errors_map_traj else None
            ),
            "mean_rot_motion_err_map_traj": (
                float(np.mean(rot_motion_errs_map_traj))
                if has_rot_motion_errors_map_traj else None
            ),
            "rmse_rot_motion_err_map_traj": (
                float(np.sqrt(np.mean(np.square(rot_motion_errs_map_traj))))
                if has_rot_motion_errors_map_traj else None
            ),
            "worst_rot_motion_err_map_traj": (
                float(np.max(rot_motion_errs_map_traj))
                if has_rot_motion_errors_map_traj else None
            ),
            "p90_rot_motion_err_map_traj": (
                float(np.percentile(rot_motion_errs_map_traj, 90))
                if has_rot_motion_errors_map_traj else None
            ),
            "rate_above_thres_trans_err_map_traj": (
                float(np.mean(trans_errs_map_traj > TRANS_ERRS_MAP_TRAJ_THRES))
                if has_map_trajectory_errors else None
            ),
            "rate_above_thres_rot_err_map_traj": (
                float(np.mean(rot_errs_map_traj > ROT_ERRS_MAP_TRAJ_THRES))
                if has_map_trajectory_errors else None
            ),
            "mean_trans_err_map_traj_impr_over_raw_odom": (
                float(np.mean(trans_errs_map_traj_impr_over_raw_odom))
                if has_trans_map_traj_impr_over_raw_odom  else None
            ),
            "median_trans_err_map_traj_impr_over_raw_odom": (
                float(np.median(trans_errs_map_traj_impr_over_raw_odom))
                if has_trans_map_traj_impr_over_raw_odom  else None
            ),
            "perc_10_trans_err_map_traj_impr_over_raw_odom": (
                float(np.percentile(trans_errs_map_traj_impr_over_raw_odom, 10))
                if has_trans_map_traj_impr_over_raw_odom  else None
            ),
            "mean_rot_err_map_traj_impr_over_raw_odom": (
                float(np.mean(rot_errs_map_traj_impr_over_raw_odom))
                if has_rot_map_traj_impr_over_raw_odom  else None
            ),
            "median_rot_err_map_traj_impr_over_raw_odom": (
                float(np.median(rot_errs_map_traj_impr_over_raw_odom))
                if has_rot_map_traj_impr_over_raw_odom  else None
            ),
            "perc_10_rot_err_map_traj_impr_over_raw_odom": (
                float(np.percentile(rot_errs_map_traj_impr_over_raw_odom, 10))
                if has_rot_map_traj_impr_over_raw_odom  else None
            ),
            
            # Compute slopes for final MAP trajectory
            "mean_pos_trans_err_slopes_map_traj": (
                float(np.mean(trans_err_slopes_map_traj))
                if has_pos_slope_trans_map_traj else None
            ),
            "rmse_pos_trans_err_slopes_map_traj": (
                float(np.sqrt(np.mean(np.square(trans_err_slopes_map_traj))))
                if has_pos_slope_trans_map_traj else None
            ),
            "p90_pos_trans_err_slopes_map_traj": (
                float(np.percentile(trans_err_slopes_map_traj, 90))
                if has_pos_slope_trans_map_traj else None
            ),
            "mean_pos_rot_err_slopes_map_traj": (
                float(np.mean(rot_err_slopes_map_traj))
                if has_pos_slope_rot_map_traj else None
            ),
            "rmse_pos_rot_err_slopes_map_traj": (
                float(np.sqrt(np.mean(np.square(rot_err_slopes_map_traj))))
                if has_pos_slope_rot_map_traj else None
            ),
            "p90_pos_rot_err_slopes_map_traj": (
                float(np.percentile(rot_err_slopes_map_traj, 90))
                if has_pos_slope_rot_map_traj else None
            ),

            # Analyze closeset p before and after resampling when only when resampling took place
            # ________________________________________________________________________________________
            "mean_trans_closest_p_after_before_resamp_vals": (
                float(np.mean(trans_closest_p_after_before_resamp_vals))
                if has_trans_closest_p_after_before_resamp_vals else None
            ),
            "worst_trans_closest_p_after_before_resamp_vals": (
                float(np.max(trans_closest_p_after_before_resamp_vals))
                if has_trans_closest_p_after_before_resamp_vals else None
            ),
            "mean_rot_closest_p_after_before_resamp_vals": (
                float(np.mean(rot_closest_p_after_before_resamp_vals))
                if has_rot_closest_p_after_before_resamp_vals else None
            ),
            "worst_rot_closest_p_after_before_resamp_vals": (
                float(np.max(rot_closest_p_after_before_resamp_vals))
                if has_rot_closest_p_after_before_resamp_vals else None
            ),

            # Compute resampling infos
            # ________________________________________________________________________________________
            "resampling_count": int(resampling_count),
            "resampling_rate": (
                float(resampling_count) / particle_update_counter if particle_update_counter else None
            ),
            "best_p_is_closest_before_resamp_rate": (
                float(best_p_is_closest_before_resamp_count) / particle_update_counter if particle_update_counter else None
            ),
            "mean_unique_resampled_parents": (
                float(np.mean(unique_resampled_parents_values))
                if has_unique_resampled_parents_values else None
            ),
            "min_unique_resampled_parents": (
                float(np.min(unique_resampled_parents_values))
                if has_unique_resampled_parents_values else None
            ),
            "mean_neff": (
                float(np.mean(neff_values))
                if has_neff_values else None
            ),
            "min_neff": (
                float(np.min(neff_values))
                if has_neff_values else None
            ),
            "max_neff": (
                float(np.max(neff_values))
                if has_neff_values else None
            ),
            "final_neff": (
                float(neff_values[-1])
                if has_neff_values else None
            ),
            "mean_neff_ratio": (
                float(np.mean(neff_ratio_values))
                if has_neff_ratio_values else None
            ),
            "min_neff_ratio": (
                float(np.min(neff_ratio_values))
                if has_neff_ratio_values else None
            ),
            "final_neff_ratio": (
                float(neff_ratio_values[-1])
                if has_neff_ratio_values else None
            )
        }

        return summary



def test():
    # step_res.resampling = particle_inherit_indices is not None
    arr = [True, True, True]

    arr_flaot = np.asarray(arr, dtype=float)
    print(f"Arr before conversion to float: {arr}")
    print(f"Arr after conversion to float: {arr_flaot}")



def test_restore_trajectory():
    # Define test data
    # Particle weights 
    p_weights = np.array([
        [3.0, 1.0, 2.0],
        [3.0, 1.0, 2.0],
        [3.0, 1.0, 2.0],
        [2.0, 3.0, 1.0],
        [2.0, 3.0, 1.0],
        [2.0, 3.0, 1.0],
    ])

    particle_inherit_indices = [
        None, 
        None, 
        np.array([0, 0, 2]),
        None, 
        None,
        np.array([0, 1, 1]),
    ]

    trans_errs_before_resampling_list = [
        np.array([0.1, 0.3, 0.2]),
        np.array([0.1, 0.3, 0.2]),
        np.array([0.1, 0.3, 0.2]),
        np.array([0.2, 0.1, 0.3]),
        np.array([0.2, 0.1, 0.3]),
        np.array([0.2, 0.1, 0.3]),
    ]

    rot_errs_before_resampling_list = [
        np.array([0.1, 0.3, 0.2]),
        np.array([0.1, 0.3, 0.2]),
        np.array([0.1, 0.3, 0.2]),
        np.array([0.2, 0.1, 0.3]),
        np.array([0.2, 0.1, 0.3]),
        np.array([0.2, 0.1, 0.3]),
    ]

    # Estimate p with highest weight at the end
    best_p_idx = np.argmax(p_weights[-1])
    print(f"Best particle idx in last iteration is: {best_p_idx}")

    # Restore map trajectory errors
    trans_errs_traj, rot_errs_traj = RBPFEValMultParticles._restore_map_trajectory_errors(
        best_particle_idx=best_p_idx,
        particle_inherit_indices=particle_inherit_indices,
        trans_errs_before_resampling_list=trans_errs_before_resampling_list,
        rot_errs_before_resampling_list=rot_errs_before_resampling_list,
    )


    print(f"Translational errors of best particle trajectory:\n{trans_errs_traj}")
    print(f"Rotational errors of best particle trajectory:\n{rot_errs_traj}")
    

def test_summarize_run():
    step_results = None
    evaluator = RBPFEValMultParticles()
    summary = evaluator.summarize_run(step_results)
    print("Summary of the run:")





def main():
    test()
    # test_restore_trajectory()
    # test_summarize_run()
    


if __name__ == "__main__":
    main()
