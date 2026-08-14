from typing import Sequence, Tuple

import numpy as np
from numba import njit

from .measurement_model import MeasurementModel
from ..scan_matcher.ogm_scan_matching import OGM
from slam.infrastructure.defs import Pose2D



@njit(cache=True, nogil=True, inline="always")
def transform_point_to_grid_cell(
    x: float,
    y: float,
    shift_x: float,
    shift_y: float,
    grid_resolution: float,
) -> Tuple[int, int]:
    '''Transforms an (x, y) point to the array access indices (i, j for row, column).'''
    i = int(np.floor((y + shift_y) / grid_resolution))
    j = int(np.floor((x + shift_x) / grid_resolution))
    return i, j


@njit(cache=True, nogil=True, inline="always")
def cell_inside_map(
    i: int,
    j: int,
    n_rows: int,
    n_cols: int,
) -> bool:
    '''Return whether map cell (i, j) lies inside the map array.'''
    return 0 <= i < n_rows and 0 <= j < n_cols


@njit(cache=True, nogil=True, inline="always")
def transform_grid_cell_to_point(
    i: int,
    j: int,
    grid_resolution: float,
    shift_x: float,
    shift_y: float,
) -> Tuple[float, float]:
    '''Transform map indices (i, j) to the world-space cell center.'''
    half_resolution = 0.5 * grid_resolution
    x = j * grid_resolution - shift_x + half_resolution
    y = i * grid_resolution - shift_y + half_resolution
    return x, y


@njit(cache=True, nogil=True)
def likelihood_numba(
    x: float,
    y: float,
    theta: float,
    measurements: np.ndarray,
    log_odds_map: np.ndarray,
    grid_resolution: float,
    shift_x: float,
    shift_y: float,
    usable_range: float,
    sigma: float,
    kernel_size: int,
    fullness_threshold: float,
    free_threshold: float,
    gaussian_sigma: float,
    free_cell_ratio: float,
) -> Tuple[float, float, int]:
    '''
    Compute the single-pose GMapping likelihood in compiled Numba code.

    Parameters
    ----------
    x, y, theta : float
        Pose in world coordinates.
    measurements : np.ndarray
        The list of measurements (e.g., LiDAR scan points) to compare against the scan matcher.
    log_odds_map : np.ndarray
        Two-dimensional log-odds map indexed as [row_i, column_j].
    grid_resolution : float
        Map cell size in metres.
    shift_x, shift_y : float
        Translation used by the map's world-to-grid transformation. -shiftx, -shifty marks the bottom left position of the 
        array, representing the ogm.
    usable_range : float
        Maximum measurement range considered by the likelihood model.
    sigma : float
        Denominator used by the GMapping log-likelihood term.
    kernel_size : int
        Search radius around each measured endpoint in grid cells.
    fullness_threshold : float
        Minimum log-odds value required for a hit candidate.
    free_threshold : float
        Maximum log-odds value allowed for the preceding free candidate.
    gaussian_sigma : float
        Denominator used by the GMapping score term.
    free_cell_ratio : float
        Distance from the endpoint to the free test point, expressed in grid-cell diagonals.
    '''
    score = 0.0
    log_likelihood = 0.0
    matched_count = 0

    null_likelihood = -0.5
    no_hit = null_likelihood / sigma

    n_rows, n_cols = log_odds_map.shape
    free_delta = grid_resolution * free_cell_ratio

    for measurement_idx in range(measurements.shape[0]):
        r = measurements[measurement_idx, 0]
        bearing = measurements[measurement_idx, 1]

        # Invalid measurements do not contribute to the likelihood.
        if not np.isfinite(r) or r <= 0.0 or r > usable_range:
            continue

        angle = theta + bearing
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # Measurement endpoint in world coordinates.
        phit_x = x + r * cos_angle
        phit_y = y + r * sin_angle

        iphit_i, iphit_j = transform_point_to_grid_cell(
            phit_x,
            phit_y,
            shift_x,
            shift_y,
            grid_resolution,
        )

        # Cell immediately before the hit cell in beam direction.
        r_free = r - free_delta
        if r_free < 0.0:
            continue

        pfree_x = x + r_free * cos_angle
        pfree_y = y + r_free * sin_angle

        ipfree_i, ipfree_j = transform_point_to_grid_cell(
            pfree_x,
            pfree_y,
            shift_x,
            shift_y,
            grid_resolution,
        )

        free_offset_i = ipfree_i - iphit_i
        free_offset_j = ipfree_j - iphit_j

        found = False
        best_dist2 = 0.0

        # Search for the closest occupied candidate whose corresponding
        # beam-predecessor cell satisfies the free-cell condition.
        for di in range(-kernel_size, kernel_size + 1):
            hit_i = iphit_i + di

            for dj in range(-kernel_size, kernel_size + 1):
                hit_j = iphit_j + dj
                free_i = hit_i + free_offset_i
                free_j = hit_j + free_offset_j

                if not cell_inside_map(hit_i, hit_j, n_rows, n_cols):
                    continue

                if not cell_inside_map(free_i, free_j, n_rows, n_cols):
                    continue

                hit_log_odds = log_odds_map[hit_i, hit_j]
                free_log_odds = log_odds_map[free_i, free_j]

                if (
                    hit_log_odds > fullness_threshold
                    and free_log_odds < free_threshold
                ):
                    cell_x, cell_y = transform_grid_cell_to_point(
                        hit_i,
                        hit_j,
                        grid_resolution,
                        shift_x,
                        shift_y,
                    )

                    dx_world = phit_x - cell_x
                    dy_world = phit_y - cell_y
                    dist2 = dx_world * dx_world + dy_world * dy_world

                    if not found or dist2 < best_dist2:
                        best_dist2 = dist2
                        found = True

        if found:
            score += np.exp(-best_dist2 / gaussian_sigma)
            log_likelihood += -best_dist2 / sigma
            matched_count += 1
        else:
            log_likelihood += no_hit

    return score, log_likelihood, matched_count


@njit(cache=True, nogil=True)
def likelihood_batch_numba(
    poses: np.ndarray,
    measurements: np.ndarray,
    log_odds_map: np.ndarray,
    grid_resolution: float,
    shift_x: float,
    shift_y: float,
    usable_range: float,
    sigma: float,
    kernel_size: int,
    fullness_threshold: float,
    free_threshold: float,
    gaussian_sigma: float,
    free_cell_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute GMapping likelihood values for a batch of poses.

    The scan, map, and model parameters are shared by all poses. The complete
    pose loop remains inside compiled Numba code, so the Python wrapper incurs
    only one compiled-function call for the whole optimized-proposal batch.
    """
    n_poses = poses.shape[0]

    scores = np.empty(n_poses, dtype=np.float64)
    log_likelihoods = np.empty(n_poses, dtype=np.float64)
    matched_counts = np.empty(n_poses, dtype=np.int64)

    for pose_idx in range(n_poses):
        score, log_likelihood, matched_count = likelihood_numba(
            x=poses[pose_idx, 0],
            y=poses[pose_idx, 1],
            theta=poses[pose_idx, 2],
            measurements=measurements,
            log_odds_map=log_odds_map,
            grid_resolution=grid_resolution,
            shift_x=shift_x,
            shift_y=shift_y,
            usable_range=usable_range,
            sigma=sigma,
            kernel_size=kernel_size,
            fullness_threshold=fullness_threshold,
            free_threshold=free_threshold,
            gaussian_sigma=gaussian_sigma,
            free_cell_ratio=free_cell_ratio,
        )

        scores[pose_idx] = score
        log_likelihoods[pose_idx] = log_likelihood
        matched_counts[pose_idx] = matched_count

    return scores, log_likelihoods, matched_counts




class LikelihoodFiledModelGmapping(MeasurementModel):
    """GMapping-style likelihood field model using a raw log-odds map."""

    def __init__(self, sigma: float = 0.1) -> None:
        if sigma <= 0.0:
            raise ValueError("sigma must be greater than zero.")

        self.sigma = float(sigma)

    def likelihood(
        self,
        pose: Pose2D,
        measurements: Sequence[Tuple[float, float]],
        log_odds_map: np.ndarray,
        grid_resolution: float,
        shift_x: float,
        shift_y: float,
        usable_range: float,
        kernel_size: int = 1,
        fullness_threshold: float = 1.2,
        free_threshold: float = 1.2,
        gaussian_sigma: float = 0.05,
        free_cell_ratio: float = np.sqrt(2.0),
    ) -> Tuple[float, float, int]:
        """
        Compute the GMapping likelihood for one pose.

        This method is only a Python wrapper. The complete beam processing and
        neighborhood search are performed by :func:`likelihood_numba`.

        Parameters
        ----------
        pose:
            Robot pose (x, y, theta) in world coordinates.
        measurements:
            Laser measurements as (range, bearing) pairs.
        log_odds_map:
            Two-dimensional log-odds map indexed as [row_i, column_j].
        grid_resolution:
            Map cell size in metres.
        shift_x, shift_y:
            Translation used by the map's world-to-grid transformation.
        usable_range:
            Maximum measurement range considered by the likelihood model.
        kernel_size:
            Search radius around each measured endpoint in grid cells.
        fullness_threshold:
            Minimum log-odds value required for a hit candidate.
        free_threshold:
            Maximum log-odds value allowed for the preceding free candidate.
        gaussian_sigma:
            Denominator used by the GMapping score term.
        free_cell_ratio:
            Distance from the endpoint to the free test point, expressed in
            grid-cell diagonals.
        """
        # Ensure valid input parameters
        if grid_resolution <= 0.0:
            raise ValueError("grid_resolution must be greater than zero.")

        if usable_range <= 0.0:
            raise ValueError("usable_range must be greater than zero.")

        if kernel_size < 0:
            raise ValueError("kernel_size must be greater than or equal to zero.")

        if gaussian_sigma <= 0.0:
            raise ValueError("gaussian_sigma must be greater than zero.")

        if free_cell_ratio < 0.0:
            raise ValueError("free_cell_ratio must be greater than or equal to zero.")

        map_array = np.asarray(log_odds_map)
        if map_array.ndim != 2:
            raise ValueError("log_odds_map must be a two-dimensional array.")

        measurements_array = np.asarray(measurements, dtype=np.float64)
        if measurements_array.size == 0:
            return 0.0, 0.0, 0

        if measurements_array.ndim != 2 or measurements_array.shape[1] != 2:
            raise ValueError("measurements must have shape (N, 2).")

        # Extract pose
        x, y, theta = pose

        # Compute the meas likelihood using the Numba-compiled function
        return likelihood_numba(
            x=float(x),
            y=float(y),
            theta=float(theta),
            measurements=measurements_array,
            log_odds_map=map_array,
            grid_resolution=float(grid_resolution),
            shift_x=float(shift_x),
            shift_y=float(shift_y),
            usable_range=float(usable_range),
            sigma=self.sigma,
            kernel_size=int(kernel_size),
            fullness_threshold=float(fullness_threshold),
            free_threshold=float(free_threshold),
            gaussian_sigma=float(gaussian_sigma),
            free_cell_ratio=float(free_cell_ratio),
        )


    def likelihood_batch(
        self,
        poses: np.ndarray,
        measurements: Sequence[Tuple[float, float]],
        log_odds_map: np.ndarray,
        grid_resolution: float,
        shift_x: float,
        shift_y: float,
        usable_range: float,
        kernel_size: int = 1,
        fullness_threshold: float = 1.2,
        free_threshold: float = 1.2,
        gaussian_sigma: float = 0.05,
        free_cell_ratio: float = np.sqrt(2.0),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute GMapping likelihood values for multiple sampled poses.

        This is the batch counterpart of :meth:`likelihood`, intended for the
        sampled ``x_j`` poses of the RBPF optimized proposal. The measurements,
        map, and model parameters are converted once and shared by all poses.
        The returned arrays preserve the input pose order.

        Parameters
        ----------
        poses:
            Pose array with shape ``(N, 3)`` and columns ``(x, y, theta)``.
        measurements:
            Shared laser measurements as ``(range, bearing)`` pairs.
        log_odds_map:
            Two-dimensional log-odds map indexed as ``[row_i, column_j]``.
        grid_resolution:
            Map cell size in metres.
        shift_x, shift_y:
            Translation used by the map's world-to-grid transformation.
        usable_range:
            Maximum measurement range considered by the likelihood model.
        kernel_size:
            Search radius around each measured endpoint in grid cells.
        fullness_threshold:
            Minimum log-odds value required for a hit candidate.
        free_threshold:
            Maximum log-odds value allowed for the preceding free candidate.
        gaussian_sigma:
            Denominator used by the GMapping score term.
        free_cell_ratio:
            Distance from the endpoint to the free test point, expressed in
            grid-cell diagonals.

        Returns
        -------
        scores, log_likelihoods, matched_counts:
            One value per input pose. ``scores`` and ``log_likelihoods`` use
            ``float64``; ``matched_counts`` uses ``int64``.
        """
        if grid_resolution <= 0.0:
            raise ValueError("grid_resolution must be greater than zero.")

        if usable_range <= 0.0:
            raise ValueError("usable_range must be greater than zero.")

        if kernel_size < 0:
            raise ValueError("kernel_size must be greater than or equal to zero.")

        if gaussian_sigma <= 0.0:
            raise ValueError("gaussian_sigma must be greater than zero.")

        if free_cell_ratio < 0.0:
            raise ValueError(
                "free_cell_ratio must be greater than or equal to zero."
            )

        poses_array = np.ascontiguousarray(poses, dtype=np.float64)
        if poses_array.ndim != 2 or poses_array.shape[1] != 3:
            raise ValueError("poses must have shape (N, 3).")

        map_array = np.ascontiguousarray(log_odds_map)
        if map_array.ndim != 2:
            raise ValueError("log_odds_map must be a two-dimensional array.")

        measurements_array = np.ascontiguousarray(
            measurements,
            dtype=np.float64,
        )
        if measurements_array.size == 0:
            n_poses = poses_array.shape[0]
            return (
                np.zeros(n_poses, dtype=np.float64),
                np.zeros(n_poses, dtype=np.float64),
                np.zeros(n_poses, dtype=np.int64),
            )

        if measurements_array.ndim != 2 or measurements_array.shape[1] != 2:
            raise ValueError("measurements must have shape (N, 2).")

        return likelihood_batch_numba(
            poses=poses_array,
            measurements=measurements_array,
            log_odds_map=map_array,
            grid_resolution=float(grid_resolution),
            shift_x=float(shift_x),
            shift_y=float(shift_y),
            usable_range=float(usable_range),
            sigma=self.sigma,
            kernel_size=int(kernel_size),
            fullness_threshold=float(fullness_threshold),
            free_threshold=float(free_threshold),
            gaussian_sigma=float(gaussian_sigma),
            free_cell_ratio=float(free_cell_ratio),
        )
