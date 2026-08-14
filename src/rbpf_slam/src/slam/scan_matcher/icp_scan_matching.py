#!/usr/bin/env python3

from __future__ import annotations

import time
from typing import Tuple, List, Optional

from numba import njit, prange
import numpy as np

from dataclasses import dataclass

import matplotlib.pyplot as plt
from math import sin, cos, atan2, pi, inf
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree

from heapq import heappush, heappop



@dataclass
class ICPResult:
    transformation: np.ndarray      # (3,1)
    use_transformation: bool
    reason: str
    mean_error: float
    n_iterations: int
    n_correspondences: int



@njit(cache=True, nogil=True)
def compute_normals_numba(points, indices):
    '''
    Get's a pointcloud and computes the normals for each point by finding the k nearest neighbors for each point
    and computing the direction of the lowest variance for that points by PCA. This direction is the normal direction, 
    the direction perpendicular to the local surface.

    This fixed variant adds a robust fallback for degenerate neighborhoods where the computed eigenvector
    has near-zero length.

    Parameters
    ----------
    points: np.ndarray
        Nx2 numpy array of points.
    indices: np.ndarray
        Nxk numpy array of indices of the k nearest neighbors for each point.
    
    Returns 
    ----------
    normals: np.ndarray
        Nx2 numpy array of normal vectors corresponding to each point.

    '''
    n_points = points.shape[0]
    k = indices.shape[1]

    # Initialize normals array
    normals = np.zeros((n_points, 2))

    for i in range(n_points):
        # Compute center of mass for nearest neighbors
        cx = 0.0
        cy = 0.0
        for j in range(k):
            idx = indices[i, j]
            cx += points[idx, 0]
            cy += points[idx, 1]
        cx /= k
        cy /= k

        # Init covariance matrix elements
        c00 = 0.0
        c01 = 0.0
        c11 = 0.0

        # Compute covariance matrix
        for j in range(k):
            # Center NN around the mean of the NN
            idx = indices[i, j]
            dx = points[idx, 0] - cx
            dy = points[idx, 1] - cy

            # Accumulate covariance matrix elements
            c00 += dx * dx
            c01 += dx * dy
            c11 += dy * dy

        # Estimate eigenvalues by compiuting determinant and find lambda values in quadratic function
        lambda_coeff = c00 + c11
        quaqdratic_const = c00 * c11 - c01 * c01

        # Solve quadratic equation: lamb^2 - (c00+c11)*lambda + (c00 * c11 - c01^2) = 0
        disc = lambda_coeff * lambda_coeff * 0.25 - quaqdratic_const

        # Clamp tiny negative values due to numerical precision
        if disc < 0.0:
            disc = 0.0

        tmp = np.sqrt(disc)
        
        # Estimate min eigenvalue) -> min eigenvector (corresponding to direction of smallest variance in data)
        lambda_min = lambda_coeff * 0.5 - tmp

        # Compute eigenvector corresponding to smallest eigenvalue (normal direction)
        vx = c01
        vy = lambda_min - c00

        norm = np.sqrt(vx * vx + vy * vy)

        # Robust fallback for axis-aligned / degenerate cases
        if norm <= 1e-12:
            if c00 <= c11:
                vx = 1.0
                vy = 0.0
            else:
                vx = 0.0
                vy = 1.0
            norm = 1.0

        normals[i, 0] = vx / norm
        normals[i, 1] = vy / norm

    return normals


# TODO: Delte this, no speedup gained. Also adapt the initalization function and erase it from there!
@njit(cache=True, nogil=True)
def prepare_system_point_to_plane_numba(
    transformation_parameter: np.ndarray,
    latest_new_data: np.ndarray,
    true_data_pointpairs: np.ndarray,
    correspondences: np.ndarray,
    true_data_normals: np.ndarray,
    epsilon: float = 1e-9,
):
    """
    Prepare the weighted point-to-plane least-squares system used by ICP.

    For every valid correspondence, this function computes the point-to-plane
    residual and its Jacobian. It then accumulates:

        H = sum(weight * J.T @ J)
        g = sum(weight * J.T * residual)

    The system can later be solved using:

        dtransformation = np.linalg.lstsq(H, -g, rcond=None)[0]

    Parameters
    ----------
    transformation_parameter : np.ndarray
        Current ICP transformation parameters. Expected shape is either
        (3,) or (3, 1), containing:

            [translation_x, translation_y, rotation_theta]

        Only the rotation angle is required for the Jacobian calculation.

    latest_new_data : np.ndarray
        Array of shape (N, 2) containing the currently transformed scan points.

    true_data_pointpairs : np.ndarray
        Array of shape (M, 2) containing the reference/map points.

    correspondences : np.ndarray
        Integer array of shape (K, 2). Each row contains:

            [index_in_latest_new_data, index_in_true_data_pointpairs]

    true_data_normals : np.ndarray
        Array of shape (M, 2) containing the normal vector associated with
        every reference/map point.

    epsilon : float
        Minimum allowed squared normal magnitude. Normals smaller than this
        threshold are ignored.

    Returns
    -------
    H : np.ndarray
        Hessian approximation with shape (3, 3).

    g : np.ndarray
        Gradient vector with shape (3, 1).

    squared_error : float
        Sum of squared point-to-plane residuals over all valid correspondences.

    Notes
    -----
    Invalid correspondences are skipped when:

    - an index is outside the valid array range;
    - a scan point contains NaN or infinity;
    - a map point contains NaN or infinity;
    - a normal contains NaN or infinity;
    - a normal has near-zero magnitude;
    - the computed residual or Jacobian is non-finite.
    """

    # Initialize the Hessian, gradient and accumulated error.
    H = np.zeros((3, 3), dtype=np.float64)
    g = np.zeros((3, 1), dtype=np.float64)
    squared_error = 0.0

    n_correspondences = correspondences.shape[0]

    if n_correspondences == 0:
        return H, g, squared_error

    # Support both transformation shapes: (3,) and (3, 1).
    if transformation_parameter.ndim == 1:
        theta = transformation_parameter[2]
    else:
        theta = transformation_parameter[2, 0]

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    n_new_points = latest_new_data.shape[0]
    n_true_points = true_data_pointpairs.shape[0]
    n_normals = true_data_normals.shape[0]

    # Process every correspondence independently.
    for correspondence_idx in range(n_correspondences):

        # Extract correspondence indices.
        i = correspondences[correspondence_idx, 0]
        j = correspondences[correspondence_idx, 1]

        # Protect against invalid indices.
        if i < 0 or i >= n_new_points:
            continue

        if j < 0 or j >= n_true_points or j >= n_normals:
            continue

        # Current transformed scan point.
        x = latest_new_data[i, 0]
        y = latest_new_data[i, 1]

        # Corresponding reference/map point.
        true_x = true_data_pointpairs[j, 0]
        true_y = true_data_pointpairs[j, 1]

        # Surface normal of the reference point.
        normal_x = true_data_normals[j, 0]
        normal_y = true_data_normals[j, 1]

        # Skip non-finite input values.
        if not (
            np.isfinite(x)
            and np.isfinite(y)
            and np.isfinite(true_x)
            and np.isfinite(true_y)
            and np.isfinite(normal_x)
            and np.isfinite(normal_y)
        ):
            continue

        # Reject zero or nearly-zero normal vectors.
        normal_squared_norm = (
            normal_x * normal_x
            + normal_y * normal_y
        )

        if normal_squared_norm <= epsilon * epsilon:
            continue

        # Difference between transformed scan point and map point.
        diff_x = x - true_x
        diff_y = y - true_y

        # Point-to-plane residual:
        #
        #     e = n.T @ (scan_point - map_point)
        #
        normal_error = (
            normal_x * diff_x
            + normal_y * diff_y
        )

        if not np.isfinite(normal_error):
            continue

        # Robust correspondence weight.
        weight = 1.0 / (1.0 + normal_error * normal_error)

        # Jacobian with respect to [tx, ty, theta].
        #
        # J = [normal_x, normal_y, d_error / d_theta]
        #
        jacobian_0 = normal_x
        jacobian_1 = normal_y

        jacobian_2 = (
            normal_x * (-x * sin_theta - y * cos_theta)
            + normal_y * (x * cos_theta - y * sin_theta)
        )

        if not np.isfinite(jacobian_2):
            continue

        # Accumulate the symmetric Hessian:
        #
        #     H += weight * J.T @ J
        #
        H[0, 0] += weight * jacobian_0 * jacobian_0
        H[0, 1] += weight * jacobian_0 * jacobian_1
        H[0, 2] += weight * jacobian_0 * jacobian_2

        H[1, 1] += weight * jacobian_1 * jacobian_1
        H[1, 2] += weight * jacobian_1 * jacobian_2

        H[2, 2] += weight * jacobian_2 * jacobian_2

        # Accumulate the gradient:
        #
        #     g += weight * J.T * residual
        #
        weighted_error = weight * normal_error

        g[0, 0] += jacobian_0 * weighted_error
        g[1, 0] += jacobian_1 * weighted_error
        g[2, 0] += jacobian_2 * weighted_error

        # Accumulate the unweighted squared residual, equivalent to the
        # previous vectorized implementation.
        squared_error += normal_error * normal_error

    # Fill the lower half because the Hessian is symmetric.
    H[1, 0] = H[0, 1]
    H[2, 0] = H[0, 2]
    H[2, 1] = H[1, 2]

    return H, g, squared_error



class ICPStopCondition:
    '''
    Class that checks if ICP should stop based on multiple criteria:
    - Max iterations
    - Relative improvement
    - No improvement limit
    - Absolute error threshold
    - Transformation magnitude threshold
    '''
    def __init__(
        self,
        max_iterations: int = 10,       # max number of iterations for icp
        epsilon_rel: float = 1e-3,      # minimum relative improvement to continue icp
        no_improvement_limit: int = 2,  # maximum number of iterations without improvement
        min_error: float = 1.0,         # minimum mean error threshold to continue icp
        min_dtrans: float = 1e-4,       # minimum translation change threshold to continue icp
        min_drot: float = 1e-1          # minimum rotation change threshold to continue icp (in radians)

    ):
        # Init params
        self.max_iterations = max_iterations
        self.epsilon_rel = epsilon_rel
        self.no_improvement_limit = no_improvement_limit
        self.min_error = min_error
        self.min_dtrans = min_dtrans
        self.min_drot = min_drot

        # internal state
        self.info: dict = {}
        self.prev_error = inf
        self.mean_err = inf
        self.no_improvement_counter = 0
        self.iteration = 0
        self.dtrans_norm = None
        self.drot_abs = None
        self.stop_reason = None
        self.rel_improvement = None
        self.no_improvement_counter = 0


    def get_stop_reason(self) -> str:
        '''
        Returns the reason why ICP stopped.'''
        return self.stop_reason


    def store_info(self):
        '''
        Stores the current state of the ICP stop condition and other relevant information in the 'info' member variable.
        '''
        self.info["iteration"] = self.iteration
        self.info["mean_err"] = self.mean_err
        self.info["rel_improvement"] = self.rel_improvement
        self.info["no_improvement_counter"] = self.no_improvement_counter
        self.info["min_mean_err"] = self.min_error
        self.info["dtrans_norm"] = self.dtrans_norm
        self.info["drot_abs"] = self.drot_abs
        self.info["stop_reason"] = self.stop_reason


    def get_info(self) -> dict:
        '''
        Returns a dictionary with the current state of the stop condition. 

        Returns:
        ----------
            dict: With the following structure:
            {
                "iteration": int,
                "mean_err": float,
                "rel_improvement": float,
                "no_improvement_counter": int,
                "dtrans_norm": float,
                "drot_abs": float,
                "stop_reason": str
            }
        '''
        return self.info


    def reset(self):
        '''
        Resets the stop condition state for a new ICP run.
        '''
        self.prev_error = inf
        self.mean_err = inf
        self.no_improvement_counter = 0
        self.iteration = 0
        self.dtrans_norm = None
        self.drot_abs = None
        self.stop_reason = None
        self.rel_improvement = None
        self.no_improvement_counter = 0


    def stop_icp(self, mean_err: float, dtransformation: np.ndarray) -> bool:
        """
        Checks if ICP should stop based on the current error and transformation change.

        Parameters:
        ----------
            mean_err: The current mean error metric (e.g., mean point to plane distance between cleaned correspondences).
            dtransformation: The change in transformation (translation and rotation) from the last iteration.
        
        Returns:
        ----------
            bool: True if ICP should stop, False otherwise.
        """
        no_improvement = False 
        self.mean_err = mean_err
        stop_icp_ = False

        if self.iteration > 0:
            # Stop cause: Max iterations
            if self.iteration >= self.max_iterations:
                self.stop_reason = "Max iterations reached"
                stop_icp_ = True 

            # Stop cause: Absolute error threshold
            if self.mean_err < self.min_error:
                self.stop_reason = "Absolute error threshold reached"
                stop_icp_ = True   

            # Check transformation magnitude
            if dtransformation is not None:
                
                if not np.all(np.isfinite(dtransformation)):
                    self.stop_reason = "Non-finite transformation detected"
                    stop_icp_ = True   
                
                # Compute translation magnitude
                self.dtrans_norm = np.linalg.norm(dtransformation[:2])
                self.drot_abs = abs(dtransformation[2])[0]
                
                if self.dtrans_norm < self.min_dtrans and self.drot_abs < self.min_drot:
                    self.stop_reason = "Transformation magnitude below threshold"
                    stop_icp_ = True   

            # Compute Relative improvement
            if not np.isfinite(self.prev_error):
                self.rel_improvement = float("inf")
            else:
                self.rel_improvement = abs(self.prev_error - self.mean_err) / max(self.prev_error, 1e-12)

            # Track improvement
            if self.rel_improvement < self.epsilon_rel:
                no_improvement = True

            # Divergence check 
            if self.mean_err > self.prev_error:
                no_improvement = True

            # Update improvement counter
            if no_improvement:
                self.no_improvement_counter += 1
            else:
                self.no_improvement_counter = 0

            # Check if improvement 
            if self.no_improvement_counter >= self.no_improvement_limit:
                self.stop_reason = "No improvement limit reached"
                stop_icp_ = True   
        
        # update state
        self.prev_error = self.mean_err

        if not stop_icp_:
            self.iteration += 1

        return stop_icp_
    


class IterativeClosestPoint():
    '''
    Implements the point-to-plane ICP algorithm for scan matching. The main method is 'find_transformation', which
    takes two point clouds as input and returns the transformation parameters that best align the new data points to
    the true data points. The class also includes methods for outlier rejection, normal computation, and information
    storage for analysis and debugging.
    '''
    IDX_X= 0
    IDX_Y= 1
    IDX_THETA= 2
    MIN_POINTS = 3
    EPSILON = 1e-9
    LEGACY_REASON_COUNTER_MAP = {
        "Too few input points": "count_too_few_points",
        "Too few correspondences in first iteration": "count_too_few_corresp",
        "Too few correspondences": "count_too_few_corresp",
        "Non-finite H or g": "infinite_h_or_g",
        "Ill-conditioned Hessian": "ill_cond_H",
        "Non-finite transformation update": "infinite_dtransform",
        "Infinite mean error": "infinite_mean_err",
        "Best Transformation too large": "best_transf_too_large",
        "Best mean error too large": "best_mean_err_too_large",
    }

    def __init__(
        self, 
        stop_params: dict, 
        max_n_points:int=800, 
        max_correspondence_distance= 2.0, 
        n_neighbors: int = 10,
        skip_subsampling: bool = False,
    ):
        '''
        Initializes the ICP scan matcher with the given stop parameters and maximum correspondence distance.

        Parameters:
        ----------
            stop_params: A dictionary containing the parameters for the ICP stop condition. This includes:
                - max_iterations: Maximum number of ICP iterations before stopping.
                - epsilon_rel: Minimum relative improvement required to continue ICP.
                - no_improvement_limit: Number of consecutive iterations with insufficient improvement before stopping.
                - min_error: Minimum mean error threshold for stopping.
                - min_dtrans: Minimum translation change required to continue ICP.
                - min_drot: Minimum rotation change required to continue ICP.
            max_correspondence_distance: The maximum distance between corresponding points to be considered in the ICP
            neighbors_pca: The number of neighbors to use for PCA when computing normals.
    
        '''    
        # Init NN
        self.neighbor= NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree')        
        
        # Init params
        # Max dist for correspondences. All correspondences with a bigger distance will be rejected as outliers
        self.max_correspond_dist= max_correspondence_distance
        # Number of neighbors to use for PCA when computing normals
        self.neighbors = n_neighbors
        # The true pointclous data will be subsampled to this amount, in every run (before outlier rejection, etc) 
        self.max_n_points = max_n_points

        # Init points count
        self.n_points_true_data = None
        self.n_points_new_data = None

        self.n_points_true_after_spatial_downsampling = None
        self.n_points_true_after_subsampling = None

        # Init stop condition
        self.stop_condition = ICPStopCondition(
            max_iterations=stop_params.get("max_iterations", 10),
            epsilon_rel=stop_params.get("epsilon_rel", 1e-3),
            no_improvement_limit=stop_params.get("no_improvement_limit", 2),
            min_error=stop_params.get("min_error", 1.0),
            min_dtrans=stop_params.get("min_dtrans", 1e-4),
            min_drot=stop_params.get("min_drot", 1e-1)
        )

        self.skip_subsampling = skip_subsampling

        # Minimum data requirements (defaulted if not provided in stop_params)
        self.min_points = int(stop_params.get("min_points", 20))
        self.min_corresp = int(stop_params.get("min_corresp", 15))

        # ICP safety thresholds (defaulted if not provided in stop_params)
        self.min_hessian_rank = int(stop_params.get("min_hessian_rank", 3))
        self.max_hessian_condition = float(stop_params.get("max_hessian_condition", 1e8))
        self.max_translation_jump = float(stop_params.get("max_translation_jump", 0.3))
        self.max_rotation_jump = float(stop_params.get("max_rotation_jump", np.deg2rad(60.0)))
        self.max_acceptable_mean_error = float(
            stop_params.get("max_acceptable_mean_error", self.stop_condition.min_error * 5.0)
        )
        self.downsample_grid_size = float(stop_params.get("downsample_grid_size", 0.1))

        # Add info storage
        self.info: dict = {}
        self.reason_counts: dict = {}
        self.legacy_counters = {
            "count_too_few_points": 0,
            "count_too_few_corresp": 0,
            "infinite_h_or_g": 0,
            "ill_cond_H": 0,
            "infinite_dtransform": 0,
            "infinite_mean_err": 0,
            "best_transf_too_large": 0,
            "best_mean_err_too_large": 0,
        }
        self.last_result_reason = None    

        # Add time measurements
        self.t_init_icp_transf = 0.0
        self.t_init_and_train_nn_tree_normals = 0.0
        self.t_downsampling_pointcloud = 0.0
        self.t_compute_normals = 0.0
        self.t_outlier_rejection = 0.0
        self.t_find_nn_outlier_rejec = 0.0 
        self.t_prepare_system = 0.0
        self.t_solve_least_squares = 0.0
        self.t_transf_update_and_results = 0.0
        self.t_find_trans = 0.0

        self.count_outlier_rejec = 0
        self.count_t_find_nn_outlier_rejec = 0
        self.count_prepare_system = 0
        self.count_solve_least_squares = 0
        self.count_transf_update_and_results = 0

        # Store infos in members
        self.transformed_new_data_list = []
        self.squared_error_list = []
        self.transformation_parameter_list = []
        self.list_of_cleaned_corresp = []
        self.list_of_cleaned_corresp_numb = []
        self.list_of_corresp_numb = []


    def _register_result_reason(self, reason: str) -> None:
        if not reason:
            return

        self.last_result_reason = reason
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1

        legacy_key = self.LEGACY_REASON_COUNTER_MAP.get(reason)
        if legacy_key is not None:
            self.legacy_counters[legacy_key] += 1


    def _finalize_result(self, result: ICPResult, extended: bool = True) -> ICPResult:
        # Store one termination reason per ICP run and persist aggregate counters.
        run_reason = result.reason or self.stop_condition.stop_reason
        self._register_result_reason(run_reason)
        self.store_info(extended=extended)
        self.info["best_transformation"] = np.asarray(result.transformation, dtype=float).copy()
        self.info["icp_iterations"] = int(result.n_iterations)
        self.info["icp_mean_error"] = float(result.mean_error) if result.mean_error is not None else None
        self.info["n_correspondences"] = int(result.n_correspondences)
        self.info["use_transformation"] = bool(result.use_transformation)
        self.info["stop_reason"] = run_reason
        self.stop_condition.reset()
        return result


    @staticmethod
    def _compute_mean_time(time: float, count: Optional[int] = None):
        if time is None or time <= 0.0 or count is None or count <= 0:
            return None
        return time / count
    

    @staticmethod
    def _filter_time(time: float):
        if time is None or time <= 0.0:
            return None
        return time


    def _evaluate_timings(self):
        timings = [
            self.t_init_icp_transf,
            self.t_init_and_train_nn_tree_normals,
            self.t_downsampling_pointcloud,
            self.t_compute_normals,
            self.t_outlier_rejection,
            self.t_find_nn_outlier_rejec, 
            self.t_prepare_system,
            self.t_solve_least_squares,
            self.t_transf_update_and_results,
        ]

        # Filter timings
        cleaned_timings = [t for t in timings if t is not None]

        if not cleaned_timings:
            return None
        
        # Compute total time for icp step
        self.t_find_trans = sum(cleaned_timings)

        # Compute mean timings for iteration timings
        mean_t_outlier_rejec = self._compute_mean_time(self.t_outlier_rejection, self.count_outlier_rejec)
        mean_t_find_nn_outlier_rejec = self._compute_mean_time(self.t_find_nn_outlier_rejec, self.count_t_find_nn_outlier_rejec)
        mean_t_prepare_system = self._compute_mean_time(self.t_prepare_system, self.count_prepare_system)
        mean_t_solve_least_squares = self._compute_mean_time(self.t_solve_least_squares, self.count_solve_least_squares)
        mean_t_transf_update_and_results = self._compute_mean_time(self.t_transf_update_and_results, self.count_transf_update_and_results)
        
        # Store timings
        self.info["t_init_icp_trans"] = self._filter_time(self.t_init_icp_transf)
        self.info["t_init_and_train_nn_tree_normals"] = self._filter_time(self.t_init_and_train_nn_tree_normals)
        self.info["t_downsampling_pointcloud"] = self._filter_time(self.t_downsampling_pointcloud)
        self.info["t_compute_normals"] = self._filter_time(self.t_compute_normals)
        self.info["t_outlier_rejection"] = self._filter_time(mean_t_outlier_rejec)
        self.info["t_find_nn_outlier_rejec"] = self._filter_time(mean_t_find_nn_outlier_rejec)
        self.info["t_prepare_system"] = self._filter_time(mean_t_prepare_system)
        self.info["t_solve_least_squares"] = self._filter_time(mean_t_solve_least_squares)
        self.info["t_transf_update_and_results"] = self._filter_time(mean_t_transf_update_and_results)
        # Overall timings
        self.info["t_find_trans"] = self._filter_time(self.t_find_trans)
                

    def store_info(self, extended: bool = False):
        '''
        Stores the current state of the ICP stop condition and other relevant information in the 'info' member variable.
        '''
        # Store stop condition info 
        self.stop_condition.store_info()

        # Reset icp info
        self.info = dict(self.stop_condition.get_info())

        # Update icp info
        self.info["max_correspondence_distance"] = self.max_correspond_dist
        self.info["min_points"] = self.min_points
        self.info["min_corresp"] = self.min_corresp
        self.info["min_hessian_rank"] = self.min_hessian_rank
        self.info["max_hessian_condition"] = self.max_hessian_condition
        self.info["max_translation_jump"] = self.max_translation_jump
        self.info["max_rotation_jump"] = self.max_rotation_jump
        self.info["max_acceptable_mean_error"] = self.max_acceptable_mean_error
        self.info["downsample_grid_size"] = self.downsample_grid_size
        self.info["n_points_true_data"] = self.n_points_true_data
        self.info["n_points_new_data"] = self.n_points_new_data
        self.info["n_points_true_after_subsampling"] = self.n_points_true_after_subsampling
        self.info["n_points_true_after_spatial_downsampling"] = self.n_points_true_after_spatial_downsampling
        self.info["last_result_reason"] = self.last_result_reason
        self.info["stop_reason_counts"] = dict(self.reason_counts)

        # Time measurements
        self._evaluate_timings()

        # Add legacy counters to info  
        for key, value in self.legacy_counters.items():
            self.info[key] = value

        if extended:
            self.info["transformed_new_data_list"] = self.transformed_new_data_list
            self.info["squared_error_list"] = self.squared_error_list
            self.info["transformation_parameter_list"] = self.transformation_parameter_list
            self.info["list_of_cleaned_corresp"] = self.list_of_cleaned_corresp
            self.info["list_of_cleaned_corresp_numb"] = self.list_of_cleaned_corresp_numb
            self.info["list_of_corresp_numb"] = self.list_of_corresp_numb


    def get_info(self) -> dict:
        '''
        Returns the current state of the ICP stop condition and other relevant information.

        Returns
        ----------
            dict: With the following structure:
            {
                "iteration": int,
                "mean_err": float,
                "rel_improvement": float,
                "no_improvement_counter": int,
                "dtrans_norm": float,
                "drot_abs": float,
                "stop_reason": str,
                "max_correspondence_distance": float,
                "min_squared_error": float,
                "n_points_true_data": int,
                "n_points_new_data": int,
                "transformed_new_data_list": List[np.ndarray],  # List of transformed new data at each iteration
                "squared_error_list": List[float],  # List of squared errors at each iteration
                "transformation_parameter_list": List[np.ndarray],  # List of transformation parameters at each iteration
                "list_of_cleaned_corresp": List[List[Tuple[int, int]]],  # List of cleaned correspondences at each iteration
                "list_of_cleaned_corresp_numb": List[int]  # List of number of cleaned correspondences at each iteration
                "list_of_corresp_numb": List[int]  # List of number of correspondences at each iteration
            }
        '''
        return self.info


    @staticmethod
    def sanitize_pointcloud(pointcloud: np.ndarray) -> np.ndarray:
        '''
        Return a finite Nx2 pointcloud array. Invalid rows are removed.
        ''' 
        pointcloud = np.asarray(pointcloud, dtype=float)

        if pointcloud.ndim != 2 or pointcloud.shape[1] != 2:
            return np.empty((0, 2), dtype=float)

        finite_rows = np.all(np.isfinite(pointcloud), axis=1)
        return pointcloud[finite_rows]


    @staticmethod
    def vec3_to_mat3(vec: np.ndarray) -> np.ndarray:
        '''
        Converts a 3D transformation vector (tx, ty, theta) to a 3x3 homogeneous transformation matrix.
        '''
        tx, ty, theta = vec.flatten()
        c = np.cos(theta)
        s = np.sin(theta)

        T = np.array([
            [c, -s, tx],
            [s, c, ty],
            [0, 0, 1]
        ])

        return T


    @staticmethod
    def mat3_to_vec3(mat: np.ndarray) -> np.ndarray:
        '''
        Converts a 3x3 homogeneous transformation matrix to a 3D transformation vector (tx, ty, theta).
        '''
        tx = mat[0, 2]
        ty = mat[1, 2]
        theta = np.arctan2(mat[1, 0], mat[0, 0])

        vec = np.array([
            [tx],
            [ty],
            [theta]
        ])

        return vec
        

    @staticmethod
    def correct_pose(pose:Tuple[float, float, float] , transf_param: np.ndarray) -> Tuple[float, float, float]:
        '''
        Corrects the given pose by the given transformation.
        '''
        # Transform transformation vector to homogeneous transformation matrix
        T = IterativeClosestPoint.vec3_to_mat3(transf_param)

        # Extract parameters
        tx, ty, rot_theta = transf_param.flatten()
        x, y, theta = pose

        # get point
        p = np.array([x, y, 1])

        # Transform point and theta
        p_new = T @ p
        theta_new = theta + rot_theta

        # normalize angle
        theta_new = atan2(sin(theta_new), cos(theta_new))

        return (p_new[0], p_new[1], theta_new)


    @staticmethod
    def compute_normals(points: np.ndarray, step: int = 1) -> np.ndarray:
        '''Gets a numpy array of points and and a step value and calculates the corresponding
        normal vectors for every point in the given array.'''
        if points.shape[0] == 0:
            return []

        normals = [np.array([0.0, 0.0])]
        visualize_normals = []
        IDX_X= 0
        IDX_Y= 1
        
        for i in range(step, points.shape[0] - step):
            # Choose points
            previous_point= points[i-step]
            current_point= points[i]
            next_point= points[i+step]
        
            # Compute normal vector
            vector= next_point - previous_point
            vector_norm = np.linalg.norm(vector)

            if not np.isfinite(vector_norm) or vector_norm <= IterativeClosestPoint.EPSILON:
                normal = np.array([0.0, 0.0])
            else:
                unit_vector= vector / vector_norm
                normal= np.array([-unit_vector[IDX_Y], unit_vector[IDX_X]])

            normals.append(normal)
        
            # For Visualization
            point_plus_normal= current_point + normal
            visualize_normal= (current_point, point_plus_normal)
            visualize_normals.append(visualize_normal)
        
        normals.append(np.array([0.0, 0.0]))
        return normals


    @staticmethod
    def compute_normals_knn_pca(points: np.ndarray, k: int = 10) -> np.ndarray:
        """
        Compute normals using KNN + PCA (2D). Find the k NN for each point. Than we find the direction of the lowest variance 
        for that points by PCA. This direction is the normal direction, the direction perpendicular to the local surface.
        
        Parameters:
        ----------
            points: Nx2 numpy array of points.
            k: Number of neighbors to use for normal estimation.
        
        Returns:
        ----------
            normals: Nx2 numpy array of normal vectors corresponding to each point.
        """
        # Check if we have enough points
        n_points = points.shape[0]
        if n_points < k:
            return np.zeros((n_points, 2))

        # init KNN and find neighbors
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(points)
        _, indices = nbrs.kneighbors(points)

        normals = np.zeros((n_points, 2))

        # For each point, compute normal based on direction with lowest variance (ICP)
        for i in range(n_points):
            # get k neighbors
            neighbors = points[indices[i]]

            # Center neighbors
            centroid = np.mean(neighbors, axis=0)
            centered = neighbors - centroid

            # Covariance matrix (2x2)
            cov = centered.T @ centered

            # Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(cov)

            # Smallest eigenvector = normal direction
            normal = eigvecs[:, 0]

            # Normalize
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                normal = normal / norm
            else:
                normal = np.array([0.0, 0.0])

            normals[i] = normal

        return normals


    def compute_normals_pca(self, points: np.ndarray, k: int = 10):
        """
        Compute normals using KNN + PCA (2D). Find the k NN for each point. Than we find the direction of the lowest variance 
        for that points by PCA. This direction is the normal direction, the direction perpendicular to the local surface.
        
        Parameters:
        ----------
            points: Nx2 numpy array of points.
            k: Number of neighbors to use for normal estimation.
        
        Returns:
        ----------
            normals: Nx2 numpy array of normal vectors corresponding to each point.
        """
        # Check if we have enough points
        n_points = points.shape[0]
        if n_points < k:
            return np.zeros((n_points, 2))
        
        _, indices = self.neighbor.kneighbors(points, n_neighbors=k)

        normals = np.zeros((n_points, 2))

        # For each point, compute normal based on direction with lowest variance (ICP)
        for i in range(n_points):
            # get k neighbors
            neighbors = points[indices[i]]

            # Center neighbors
            centroid = np.mean(neighbors, axis=0)
            centered = neighbors - centroid

            # Covariance matrix (2x2)
            cov = centered.T @ centered

            # Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(cov)

            # Smallest eigenvector = lowest variance = normal direction
            normal = eigvecs[:, 0]

            # Normalize vectors
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                normal = normal / norm
            else:
                normal = np.array([0.0, 0.0])

            normals[i] = normal

        return normals


    @staticmethod
    def compute_rotation_matrix(theta: float) -> np.ndarray:
        '''Return rotation matrix of given theta.'''
        theta = float(np.asarray(theta).item())
        return np.array([
            [cos(theta), -sin(theta)],
            [sin(theta), cos(theta)]
            ])


    def max_distance_outlier_rejection(self, new_data_points: np.ndarray, true_data_points: np.ndarray, correspondences: list) -> tuple:
        '''Get's a list of new data points and true data points and a list of (i, j) correspondences. 
        Rejects all pairs, which distance is higher than the given threshold.'''
        cleaned_correspondences= []
        sum_error= 0
        
        for i, j in correspondences:
            error= np.linalg.norm(new_data_points[i] - true_data_points[j])

            if not np.isfinite(error):
                continue

            sum_error+= error
            if error < self.max_correspond_dist:
                cleaned_correspondences.append((i, j))

        return cleaned_correspondences, sum_error


    @staticmethod
    def multiple_pairing_rejection(correspondences: list) -> list:
        '''Get's a sorted list of (j, i, distance) correspondences, sorted by j and rejects the 
        worst (j, i) correspondences, such that there is only one i that belongs to one j. One correspon-
        dence is more worse than the other, when the distance between the corresponding points is bigger, 
        than the other. returns a list of (i, j) correspondences.'''
        if not correspondences:
            return []

        # Pop first item 
        j, i, dist= heappop(correspondences)
        current_j= j

        # init all variables
        cleaned_correspondences= []
        c=(i, j)
        shortest_dist= dist

        # Search for best pairs
        for i in range(len(correspondences)):
            j, i ,dist = heappop(correspondences)
            if(j != current_j):
                current_j = j
                cleaned_correspondences.append(c)
                shortest_dist = inf
                c = (i, j)
            elif(dist < shortest_dist):
                c = (i, j)
        
        cleaned_correspondences.append(c)
        
        return cleaned_correspondences

    
    def outlier_rejection(self, new_data_points: np.ndarray, true_data_points: np.ndarray, correspondences: list) -> tuple[np.ndarray, float]:
        '''Class that uses all available methodes to reject outliers in the (j, i, distance)
        correspondences.'''
        # print("\n\nNumber Of correspondences before mpr= ", len(correspondences))
        cleaned_correspondences= self.multiple_pairing_rejection(correspondences)
        # print("Number Of correspondences after mpr= ", len(cleaned_correspondences))
        cleaned_correspondences, sum_error= self.max_distance_outlier_rejection(new_data_points, true_data_points, cleaned_correspondences)
        # print("Number Of correspondences after max_distance_outlier_rejection= ", len(cleaned_correspondences))
        return cleaned_correspondences, sum_error


    def vectorized_outlier_rejection(
        self,
        distances: np.ndarray,
        indices: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Vectorized implementation of the outlier rejection process for the icp. The outlier process consists of two steps:
            Step 1: Max distance rejection: We reject all correspondences with a distance bigger than the given threshold.
            Step 2: Multiple pairing rejection: We reject all correspondences, where one j is paired with multiple i.
                    We only keep the i which is closest to j (euclidean distance) 

        Parameters
        ----------
        distances: np.ndarray
            Nxk array of distances from each point in new_data to its k nearest neighbors in true_data.
        indices: np.ndarray
            Nxk array of indices of the k nearest neighbors in true_data for each point in new_data.
        
        Returns
        -------
        correspondences: np.ndarray
            Nx2 array of cleaned correspondences after outlier rejection.
        sum_error: float
            Sum of distances for the remaining correspondences.
        """

        # Flatten NN output
        j = indices[:, 0]          # index in true_data
        i = np.arange(len(j))      # index in new_data
        d = distances[:, 0]        # distances

        # Step 1: max distance rejection
        mask = np.isfinite(d) & (d < self.max_correspond_dist)

        i = i[mask]
        j = j[mask]
        d = d[mask]

        if len(i) == 0:
            return np.empty((0, 2), dtype=int), 0.0

        # Step 2: multiple pairing rejection (keep best per j)
        # Sort by j, then by distance
        order = np.lexsort((d, j))
        i = i[order]
        j = j[order]
        d = d[order]

        # Keep only first occurrence of each j (smallest distance due to sorting)
        unique_j, unique_indices = np.unique(j, return_index=True)

        i = i[unique_indices]
        j = j[unique_indices]
        d = d[unique_indices]

        # Compute sum error 
        sum_error = np.sum(d)

        # Return cleaned correspondences 
        correspondences = np.column_stack((i, j))

        return correspondences, sum_error


    @staticmethod
    def compute_jacobian_point_to_plane(normal: np.ndarray, theta: float, point: np.ndarray) -> np.ndarray:        
        theta = float(np.asarray(theta).item())
        x= point.item(0)
        y= point.item(1)
        x_normal= normal.item(0)
        y_normal= normal.item(1)
        third_element= x_normal * (-x*sin(theta) - y*cos(theta)) + y_normal * (x*cos(theta) - y*sin(theta))
        return np.array([[x_normal, y_normal, third_element]])


    def prepare_system_point_to_plane(
            self, 
            transformation_parameter: np.ndarray,
            latest_new_data: np.ndarray,
            true_data_pointpairs: np.ndarray,
            correspondences: list,
            true_data_normals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        '''
        Prepares the system of equations for the point-to-plane ICP algorithm. Checks if the inputs are valid and computes the Hessian
        matrix H and the gradient vector g for the least squares problem. Also computes the squared error (point to plane) for the
        current correspondences.
        '''
        # Init Hessian Matrix and gradient
        H = np.zeros((3, 3))
        g = np.zeros((3, 1))        
        squared_error= 0.0

        for i, j in correspondences:
            # Extract data
            new_data_point= latest_new_data[i]
            true_data_point= true_data_pointpairs[j]
            normal= true_data_normals[j]

            # Check if data is valid
            if not (
                np.all(np.isfinite(new_data_point)) and
                np.all(np.isfinite(true_data_point)) and
                np.all(np.isfinite(normal))
            ):
                continue

            if np.linalg.norm(normal) <= self.EPSILON:
                continue
            
            # Compute the distance error between the transformed new point and the true point
            distance_error= new_data_point - true_data_point            
            
            # Compute point to plane error by projecting the distance error onto the normal vector
            normal_error= np.dot(normal, distance_error)
            if not np.isfinite(normal_error):
                continue
            
            # : Weight the correspondence by error
            weight = 1 / (1 + normal_error**2)
            
            # Compute jacobian matrix
            J= self.compute_jacobian_point_to_plane(normal, transformation_parameter[self.IDX_THETA], new_data_point)

            if not np.all(np.isfinite(J)):
                continue
            
            # Update Hessian and gradient
            H+= weight * np.dot(J.T, J)
            g+= weight * np.dot(J.T, normal_error)   
            
            # Accumulate the squared errors
            squared_error+= normal_error**2

        return H, g, squared_error


    def prepare_system_point_to_plane_vec(
        self,
        transformation_parameter,
        latest_new_data,
        true_data_pointpairs,
        correspondences,
        true_data_normals
    ):
        '''
        Preppares the system of equations for the point-to-plane ICP algorithm in a vectorized way (faster). Method computes the gradient 
        vector g and the Hessian matrix H to minimize the point-to-plane error for the given correspondences later on.

        Parameters
        ----------
            transformation_parameter: np.ndarray
                3x1 numpy array of the current transformation parameters (tx, ty, theta).
            latest_new_data: np.ndarray
                Nx2 numpy array of the new data points after applying the current transformation.
            true_data_pointpairs: np.ndarray
                Mx2 numpy array of the true data points.
            correspondences: list
                List of (i, j) correspondences, where i is the index of the new data point and j is the index of the corresponding true data point.
            true_data_normals: np.ndarray
                Mx2 numpy array of the normal vectors corresponding to the true data points. 

        Returns
        ----------
            H: np.ndarray
                3x3 numpy array representing the Hessian matrix of the least squares problem.
            g: np.ndarray
                3x1 numpy array representing the gradient vector of the least squares problem.
            squared_error: float
                The sum of squared point-to-plane errors for the current correspondences.
        '''

        if len(correspondences) == 0:
            return np.zeros((3,3)), np.zeros((3,1)), 0.0

        # Extract indices
        i = correspondences[:, 0]
        j = correspondences[:, 1]

        # Gather data
        new_pts = latest_new_data[i]
        true_pts = true_data_pointpairs[j]
        normals = true_data_normals[j]

        # Filter finite
        mask = (
            np.all(np.isfinite(new_pts), axis=1) &
            np.all(np.isfinite(true_pts), axis=1) &
            np.all(np.isfinite(normals), axis=1)
        )

        new_pts = new_pts[mask]
        true_pts = true_pts[mask]
        normals = normals[mask]

        if new_pts.shape[0] == 0:
            return np.zeros((3,3)), np.zeros((3,1)), 0.0

        # Remove zero normals
        norm_mask = np.linalg.norm(normals, axis=1) > self.EPSILON

        new_pts = new_pts[norm_mask]
        true_pts = true_pts[norm_mask]
        normals = normals[norm_mask]

        if new_pts.shape[0] == 0:
            return np.zeros((3,3)), np.zeros((3,1)), 0.0

        # Distance error
        diff = new_pts - true_pts

        # Point-to-plane error
        normal_error = np.sum(normals * diff, axis=1)

        valid_mask = np.isfinite(normal_error)

        normal_error = normal_error[valid_mask]
        normals = normals[valid_mask]
        new_pts = new_pts[valid_mask]

        if normal_error.shape[0] == 0:
            return np.zeros((3,3)), np.zeros((3,1)), 0.0

        # Weights
        weights = 1.0 / (1.0 + normal_error**2)

        # --- Jacobian (vectorized) ---
        theta = float(transformation_parameter[self.IDX_THETA])

        x = new_pts[:, 0]
        y = new_pts[:, 1]
        nx = normals[:, 0]
        ny = normals[:, 1]

        third = nx * (-x*np.sin(theta) - y*np.cos(theta)) + \
                ny * ( x*np.cos(theta) - y*np.sin(theta))

        J = np.stack([nx, ny, third], axis=1)   # shape (N, 3)

        # Apply weights
        W = weights[:, None]

        # Hessian
        H = (J * W).T @ J

        # Gradient
        g = (J * W).T @ normal_error.reshape(-1,1)

        # Error
        squared_error = np.sum(normal_error**2)

        return H, g, squared_error


    def downsample_pointcloud_rand(self, pointcloud: np.ndarray, max_n_points: int=800) -> np.ndarray:
        '''
        Downsamples the given pointcloud to the given max number of points, randomly.
        '''
        n_points = pointcloud.shape[0]

        if n_points >= max_n_points:
            indices = np.random.choice(n_points, size=max_n_points, replace=False)
            subsampled_pointcloud = pointcloud[indices]
        else:
            subsampled_pointcloud = pointcloud
        
        return subsampled_pointcloud


    def dowmsample_pointcloud_deterministic(self, pointcloud: np.ndarray, max_n_points: int=800) -> np.ndarray:
        '''
        Downsamples the given pointcloud to the given max number of points, deterministically.
        The selected points are approximately evenly spaced over the original order.
        '''
        pointcloud = np.asarray(pointcloud, dtype=float)

        if pointcloud.ndim != 2 or pointcloud.shape[1] != 2:
            return np.empty((0, 2), dtype=float)

        n_points = pointcloud.shape[0]

        if n_points <= max_n_points:
            return pointcloud

        # Deterministic evenly spaced sampling
        indices = np.linspace(0, n_points - 1, max_n_points, dtype=int)
        subsampled_pointcloud = pointcloud[indices]

        return subsampled_pointcloud


    def downsample_pointcloud_spatial(self, pointcloud: np.ndarray, grid_size: float) -> np.ndarray:
        """
        Spatially downsamples a 2D point cloud.

        The input point cloud must have shape (N, 2), where each row is one [x, y] point.

        The method creates virtual grid cells of size grid_size in world coordinates. All points that fall into the same
        grid cell are reduced to one representative point. The representative is the first original point found in that
        grid cell.

        Parameters
        ----------
        pointcloud : np.ndarray
            Array of shape (N, 2), where N is the number of points.
        grid_size : float
            Size of one spatial grid cell in meters.

        Returns
        -------
        np.ndarray
            Downsampled point cloud of shape (M, 2), where M <= N.
        """
        # Convert points
        pointcloud = np.asarray(pointcloud, dtype=float)

        # Safety checks: valid point cloud must be (N, 2)
        if pointcloud.ndim != 2 or pointcloud.shape[1] != 2:
            return np.empty((0, 2), dtype=float)

        # Disable downsampling if grid_size is invalid or intentionally set to 0
        if grid_size is None or grid_size <= 0.0:
            return pointcloud

        n_points = pointcloud.shape[0]

        if n_points == 0:
            return pointcloud

        # Compute spatial grid index for each point
        # Example: point [1.23, 2.47], grid_size=0.10 -> grid index [12, 24]
        grid_indices = np.floor(pointcloud / grid_size).astype(np.int64)

        # Find first point index for each occupied grid cell
        _, unique_indices = np.unique(
            grid_indices,
            axis=0,
            return_index=True,
        )

        # Keep one original point per occupied spatial grid cell.
        # Sorting unique_indices restores deterministic original pointcloud order,
        # not geometric order.
        subsampled_pointcloud = pointcloud[np.sort(unique_indices)]

        return subsampled_pointcloud
    

    def find_transformation(
        self, 
        new_data_pointpairs: np.ndarray, 
        true_data_pointpairs: np.ndarray
        ) -> ICPResult:
        '''
        Get's the new data points and the true datapoints and trys to minimize the error between the two
        pointclouds by finding the best transformation. Returns the transformation parameters and stores 
        relevant information in the 'info' member variable. The info contains the data for each transformation
        run.

        Parameters:
        ----------
        new_data_pointpairs: np.ndarray
            Nx2 numpy array of the new data points.
        true_data_pointpairs: np.ndarray
            Mx2 numpy array of the true data points.
        
        Returns:
        ----------
        Result: ICPResult
            A dataclass containing:
                - transformation: 3x1 numpy array of the final transformation parameters (tx, ty, theta).
                - use_transformation: Indicator whether to use tranformation (true) or not (false). 
                - reason: str providing the reason for stopping ICP 
                - mean_error: float representing the mean error of the correspondences from the best iteration
                - n_iterations: int representing the number of ICP iterations performed.
                - n_correspondences: int representing the number of correspondences of the best iteration
        '''
        start_t_init_transf = time.perf_counter()
        # Init vars
        # TFs
        transformation = np.zeros((3, 1))
        dtransformation = transformation.copy()
        best_transformation = transformation.copy()

        # Errors
        squared_error = inf         # Squared error of ICP metric
        mean_err = inf              # Mean of the Squared error  
        best_mean_error = np.inf    # Best mean error found

        # Init info values
        self.n_points_true_after_spatial_downsampling = None
        self.n_points_true_after_subsampling = None

        # number of correspondences from best iteration
        n_corresp_best_iter = 0     

        # Init timings and counter
        self.t_init_icp_transf = 0.0
        self.t_init_and_train_nn_tree_normals = 0.0
        self.t_downsampling_pointcloud = 0.0
        self.t_compute_normals = 0.0
        self.t_outlier_rejection = 0.0
        self.t_find_nn_outlier_rejec = 0.0 
        self.t_prepare_system = 0.0
        self.t_solve_least_squares = 0.0
        self.t_transf_update_and_results = 0.0
        self.t_find_trans = 0.0
        
        self.count_outlier_rejec = 0
        self.count_t_find_nn_outlier_rejec = 0
        self.count_prepare_system = 0
        self.count_solve_least_squares = 0
        self.count_transf_update_and_results = 0

        
        # Lists to store results
        self.squared_error_list= []
        self.transformation_parameter_list = [transformation.copy()]
        self.transformed_new_data_list = [new_data_pointpairs.copy()]
        latest_new_data= new_data_pointpairs.copy()
        self.list_of_cleaned_corresp = []
        self.list_of_cleaned_corresp_numb = []
        self.list_of_corresp_numb = []
        
        # Store number of points for logging
        self.n_points_true_data = true_data_pointpairs.shape[0]
        self.n_points_new_data = new_data_pointpairs.shape[0]

        # Santize pointclouds
        new_data_pointpairs = self.sanitize_pointcloud(new_data_pointpairs)
        true_data_pointpairs = self.sanitize_pointcloud(true_data_pointpairs)
        true_pointcloud_downsampled_geometrically = None

        self.t_init_icp_transf = time.perf_counter() - start_t_init_transf

        # Check if we have enough points for icp
        if (
            new_data_pointpairs.shape[0] < self.min_points or
            true_data_pointpairs.shape[0] < self.min_points
        ):
            self.stop_condition.stop_reason = "Too few input points"
            return self._finalize_result(ICPResult(
                transformation=np.zeros((3,1)),
                use_transformation=False,
                reason="Too few input points",
                mean_error=best_mean_error,
                n_iterations=0,
                n_correspondences=0
            ), extended=True)


        # Downsampling
        start_t_downsampling = time.perf_counter()
        if not self.skip_subsampling and true_data_pointpairs.shape[0] > self.max_n_points:
            # Downsample points with geometrically relavance      
            true_pointcloud_downsampled_geometrically = self.downsample_pointcloud_spatial(
                pointcloud=true_data_pointpairs,
                grid_size=self.downsample_grid_size
            )

            # Downsample true data points
            true_data_pointpairs = self.dowmsample_pointcloud_deterministic(
                pointcloud=true_pointcloud_downsampled_geometrically,
                max_n_points=self.max_n_points
            )
            
        # get number of points after downsampling for logging
        if true_pointcloud_downsampled_geometrically is not None:
            self.n_points_true_after_spatial_downsampling = true_pointcloud_downsampled_geometrically.shape[0]

        self.n_points_true_after_subsampling = true_data_pointpairs.shape[0]

        self.t_downsampling_pointcloud = time.perf_counter() - start_t_downsampling

        if true_data_pointpairs.shape[0] < self.min_points:
            self.stop_condition.stop_reason = "Too few input points"
            return self._finalize_result(ICPResult(
                transformation=np.zeros((3,1)),
                use_transformation=False,
                reason="Too few input points",
                mean_error=best_mean_error,
                n_iterations=0,
                n_correspondences=0
            ), extended=True)

                
        # Train Nearest Neighbor with true data points 
        t_start_init_and_train_nn_tree_normals = time.perf_counter()
        n_neighbors_normals = min(self.neighbors, true_data_pointpairs.shape[0])
        self.tree = cKDTree(
            true_data_pointpairs,
            leafsize=16,
            balanced_tree=False,
            compact_nodes=False, 
            copy_data=False,
        )

        _, indices_normal = self.tree.query(
            true_data_pointpairs,
            k=n_neighbors_normals
        )

        self.t_init_and_train_nn_tree_normals = time.perf_counter() - t_start_init_and_train_nn_tree_normals
        
        # Compute normals of true data points
        start_t_compute_normals = time.perf_counter()
        true_data_normals = compute_normals_numba(
            true_data_pointpairs,
            indices_normal
        )
        self.t_compute_normals = time.perf_counter() - start_t_compute_normals

        # Reset stop condition
        self.stop_condition.reset()
                
        while True:
            # Check stop condition       
            if self.stop_condition.stop_icp(mean_err, dtransformation):
                break

            # Find Nearest Neighbor by euclidean distance
            start_t_find_nn_outlier_rejec = time.perf_counter()
            distances, corresp_new_data = self.tree.query(
                latest_new_data,
                k=1,
                # eps=0.0,
            )
            t_find_nn_outlier_rejec = time.perf_counter() - start_t_find_nn_outlier_rejec
            self.t_find_nn_outlier_rejec += t_find_nn_outlier_rejec
            self.count_t_find_nn_outlier_rejec += 1
            
            # Clean correspondences by outlier rejection 
            start_t_outlier_rejection = time.perf_counter()   
            cleaned_corresp, sum_error = self.vectorized_outlier_rejection(
                distances=distances[:, None],
                indices=corresp_new_data[:, None],
            )

            n_correspond = cleaned_corresp.shape[0]
            self.t_outlier_rejection += time.perf_counter() - start_t_outlier_rejection
            self.count_outlier_rejec += 1

            # Check if we have enough correspondences to continue
            if n_correspond < self.min_corresp:
                if self.stop_condition.iteration == 1:
                    # Return because it doesn't make sense to go on if we had not enough correspondences from beginning on. 
                    # Report the attempted iteration count and the observed correspondence count for diagnostics.
                    # mean_error stays inf because no valid ICP system was solved yet.
                    return self._finalize_result(ICPResult(
                        transformation=np.zeros((3,1)),
                        use_transformation=False,
                        reason="Too few correspondences in first iteration",
                        mean_error=np.inf,
                        n_iterations=self.stop_condition.iteration,
                        n_correspondences=n_correspond
                    ), extended=True)
                else:
                    # If we end up here might have had good correespondecnes before and wanne use those!
                    self.stop_condition.stop_reason = "Too few correspondences"
                    break            
                
            # Prepare the system
            start_t_prepare_system = time.perf_counter()
            H, g, squared_error = self.prepare_system_point_to_plane_vec(
                transformation,
                latest_new_data,
                true_data_pointpairs,
                cleaned_corresp,
                true_data_normals,
            )   

            # cleaned_corresp = np.asarray(cleaned_corresp, dtype=np.int64)
            # H, g, squared_error = prepare_system_point_to_plane_numba(
            #     transformation_parameter=transformation,
            #     latest_new_data=latest_new_data,
            #     true_data_pointpairs=true_data_pointpairs,
            #     correspondences=cleaned_corresp,
            #     true_data_normals=true_data_normals,
            # )

            self.t_prepare_system += time.perf_counter() - start_t_prepare_system
            self.count_prepare_system += 1

            # Saftey check for H and g
            if not (np.all(np.isfinite(H)) and np.all(np.isfinite(g))):
                return self._finalize_result(ICPResult(
                    transformation=best_transformation,
                    use_transformation=False,
                    reason="Non-finite H or g",
                    mean_error=best_mean_error,
                    n_iterations=self.stop_condition.iteration,
                    n_correspondences=n_corresp_best_iter
                ), extended=True)

            # Reject non-solvable systems via configurable Hessian thresholds.
            if np.linalg.matrix_rank(H) < self.min_hessian_rank or np.linalg.cond(H) > self.max_hessian_condition:
                return self._finalize_result(ICPResult(
                    transformation=best_transformation,
                    use_transformation=False,
                    reason="Ill-conditioned Hessian",
                    mean_error=best_mean_error,
                    n_iterations=self.stop_condition.iteration,
                    n_correspondences=n_corresp_best_iter
                ), extended=True)            
            

            # Compute least Squares Solution
            start_t_solve_least_squares = time.perf_counter()
            dtransformation= np.linalg.lstsq(H, -g, rcond=None)[0]
            self.t_solve_least_squares += time.perf_counter() - start_t_solve_least_squares
            self.count_solve_least_squares += 1
            
            # Safety check for dtransformation
            if not np.all(np.isfinite(dtransformation)):
                return self._finalize_result(ICPResult(
                    transformation=best_transformation,
                    use_transformation=False,
                    reason="Non-finite transformation update",
                    mean_error=best_mean_error,
                    n_iterations=self.stop_condition.iteration,
                    n_correspondences=n_corresp_best_iter
                ), extended=True)

            # Update transformation 
            # This must be done in a multiplicative way to ensure proper handling of rotations 
            start_t_transf_update_and_results = time.perf_counter()
            T = self.vec3_to_mat3(transformation)
            dT = self.vec3_to_mat3(dtransformation)
            T = dT @ T
            transformation = self.mat3_to_vec3(T)                 

            # transformation += dtransformation

            # # Normalize angles
            # transformation[self.IDX_THETA] = atan2(sin(transformation[self.IDX_THETA]), cos(transformation[self.IDX_THETA])) 
                           
            # Update rotation and translation matrix
            rotation_matrix= self.compute_rotation_matrix(transformation[self.IDX_THETA])
            translation= transformation[0:self.IDX_THETA]
            
            # Transform new data points by rotation and translation 
            latest_new_data_T= np.dot(rotation_matrix, new_data_pointpairs.T) + translation
            latest_new_data= latest_new_data_T.T

            # Compute mean err metric for stop condition check
            if np.isfinite(squared_error):
                mean_err = squared_error / n_correspond
            else:
                mean_err = inf

            # Track best iteration 
            if mean_err < best_mean_error:
                best_mean_error = mean_err
                best_transformation = transformation.copy()
                n_corresp_best_iter = n_correspond
            
            # Append data to lists
            self.transformed_new_data_list.append(latest_new_data)
            self.list_of_cleaned_corresp.append(cleaned_corresp)
            self.list_of_cleaned_corresp_numb.append(cleaned_corresp.shape[0])
            # self.list_of_corresp_numb.append(len(correspondences))
            self.squared_error_list.append(squared_error)
            self.transformation_parameter_list.append(transformation.copy())
        
        # Append last correspondence
        if self.list_of_cleaned_corresp:
            self.list_of_cleaned_corresp.append(self.list_of_cleaned_corresp[-1])
            self.list_of_cleaned_corresp_numb.append(self.list_of_cleaned_corresp_numb[-1])
            # self.list_of_corresp_numb.append(self.list_of_corresp_numb[-1])

        self.t_transf_update_and_results += time.perf_counter() - start_t_transf_update_and_results
        self.count_transf_update_and_results += 1

        # Final safety checks
        if not np.isfinite(best_mean_error):
            return self._finalize_result(ICPResult(
                transformation=best_transformation,
                use_transformation=False,
                reason="Infinite mean error",
                mean_error=best_mean_error,
                n_iterations=self.stop_condition.iteration,
                n_correspondences=n_corresp_best_iter
            ), extended=True)
    
        # Rejact large jumps in transofrmation
        trans_norm = np.linalg.norm(best_transformation[:2])
        rot = abs(best_transformation[2,0])

        # Check if translation and rotation jump are too large -> reject
        if trans_norm > self.max_translation_jump or rot > self.max_rotation_jump:
            return self._finalize_result(ICPResult(
                transformation=best_transformation,
                use_transformation=False,
                reason="Best Transformation too large",
                mean_error=best_mean_error,
                n_iterations=self.stop_condition.iteration,
                n_correspondences=n_corresp_best_iter
            ), extended=True)

        # Reject high residual error using explicit configurable threshold.
        if best_mean_error > self.max_acceptable_mean_error:
            return self._finalize_result(ICPResult(
                transformation=best_transformation,
                use_transformation=False,
                reason="Best mean error too large",
                mean_error=best_mean_error,
                n_iterations=self.stop_condition.iteration,
                n_correspondences=n_corresp_best_iter
            ), extended=True)

        return self._finalize_result(ICPResult(
            transformation=best_transformation,
            use_transformation=True,
            reason="All safety checks passed",
            mean_error=best_mean_error,
            n_iterations=self.stop_condition.iteration,
            n_correspondences=n_corresp_best_iter
        ), extended=True)

    