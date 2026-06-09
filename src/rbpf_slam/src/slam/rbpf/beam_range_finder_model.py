from typing import List, Tuple

import numpy as np
from numba import njit
from math import floor, sqrt, exp, log, pi

from .measurement_model import MeasurementModel
from slam.infrastructure.defs import Pose2D
from slam.scan_matcher.ogm_scan_matching import OGM


@njit
def _raytrace_first_occupied_cell(
    log_odds_map: np.ndarray,
    pose_i: int,
    pose_j: int,
    end_i: int,
    end_j: int,
    occ_thresh: float,
):
    """
    Bresenham raytracing from robot cell to beam endpoint.

    Returns
    -------
    found : bool
        True if an occupied cell was found along the ray.

    hit_i, hit_j : int
        First occupied cell along ray.

    out_of_map : bool
        True if ray left map before reaching endpoint.
    """

    n_rows, n_cols = log_odds_map.shape

    cell_i = pose_i
    cell_j = pose_j

    dx = abs(end_j - cell_j)
    dy = abs(end_i - cell_i)

    sx = 1 if cell_j < end_j else -1
    sy = 1 if cell_i < end_i else -1

    err = dx - dy

    first_cell = True

    while True:
        # Stop if ray leaves map
        if cell_i < 0 or cell_i >= n_rows or cell_j < 0 or cell_j >= n_cols:
            return False, -1, -1, True

        # Do not count the robot's own cell as obstacle
        if not first_cell:
            if log_odds_map[cell_i, cell_j] >= occ_thresh:
                return True, cell_i, cell_j, False

        first_cell = False

        # Stop if endpoint reached
        if cell_i == end_i and cell_j == end_j:
            break

        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            cell_j += sx

        if e2 < dx:
            err += dx
            cell_i += sy

    return False, -1, -1, False


@njit
def raytracing_log_likelihood_numba(
    log_odds_map: np.ndarray,
    measurements: np.ndarray,      # shape (N, 2): [range, bearing]
    x: float,
    y: float,
    heading: float,
    shift_x: float,
    shift_y: float,
    grid_resolution: float,
    min_sensor_range: float,
    max_sensor_range: float,
    occ_thresh: float,
    sigma_hit: float,
    z_hit: float,
    z_rand: float,
    p_max_no_obstacle: float,
    p_max_obstacle: float,
    p_no_obstacle_for_hit: float,
    beam_step: int,
):
    """
    Computes the raytracing measurement log-likelihood for one pose.

    This is a beam-endpoint / expected-ray model:

        1. For every measured beam:
           - raytrace through the map from candidate pose
           - find first occupied cell along that ray

        2. Compare:
           - measured range r_meas
           - expected map range r_expected

        3. Score with Gaussian likelihood.

    Returns
    -------
    log_likelihood : float
        Sum of log probabilities over beams.

    mean_abs_error : float
        Mean absolute range error for beams where map hit and measurement hit exist.

    valid_beam_count : int
        Number of beams used.

    map_hit_count : int
        Number of beams where raytracing found an occupied cell.

    no_map_hit_count : int
        Number of beams where raytracing found no occupied cell.

    out_of_map_count : int
        Number of rays that left the map.
    """

    n_rows, n_cols = log_odds_map.shape

    pose_i = int(floor((y + shift_y) / grid_resolution))
    pose_j = int(floor((x + shift_x) / grid_resolution))

    # Candidate pose outside map -> impossible pose
    if pose_i < 0 or pose_i >= n_rows or pose_j < 0 or pose_j >= n_cols:
        return -1.0e12, 0.0, 0, 0, 0, 0

    if beam_step < 1:
        beam_step = 1

    log_likelihood = 0.0

    valid_beam_count = 0
    map_hit_count = 0
    no_map_hit_count = 0
    out_of_map_count = 0

    abs_error_sum = 0.0
    abs_error_count = 0

    # Gaussian normalizer
    gaussian_norm = 1.0 / (sigma_hit * sqrt(2.0 * pi))

    # Small probability floor to avoid log(0)
    eps = 1.0e-12

    for k in range(0, measurements.shape[0], beam_step):

        r_meas = measurements[k, 0]
        bearing = measurements[k, 1]

        # Ignore invalid too-close beams
        if r_meas <= min_sensor_range:
            continue

        valid_beam_count += 1

        # Max-range / inf beams mean: no obstacle was measured
        measured_no_hit = False
        if (not np.isfinite(r_meas)) or r_meas >= max_sensor_range:
            measured_no_hit = True
            r_eff = max_sensor_range
        else:
            r_eff = r_meas

        # Raytrace always to max range.
        # We want to know what the map predicts as first obstacle.
        phi = heading + bearing

        ray_end_x = x + max_sensor_range * np.cos(phi)
        ray_end_y = y + max_sensor_range * np.sin(phi)

        end_i = int(floor((ray_end_y + shift_y) / grid_resolution))
        end_j = int(floor((ray_end_x + shift_x) / grid_resolution))

        found, hit_i, hit_j, out_of_map = _raytrace_first_occupied_cell(
            log_odds_map=log_odds_map,
            pose_i=pose_i,
            pose_j=pose_j,
            end_i=end_i,
            end_j=end_j,
            occ_thresh=occ_thresh,
        )

        if out_of_map:
            out_of_map_count += 1

        if found:
            map_hit_count += 1

            # Convert hit cell center to world coordinates
            hit_x = hit_j * grid_resolution - shift_x + grid_resolution / 2.0
            hit_y = hit_i * grid_resolution - shift_y + grid_resolution / 2.0

            dx = hit_x - x
            dy = hit_y - y
            r_expected = sqrt(dx * dx + dy * dy)

            if measured_no_hit:
                # Sensor says "nothing", map says "there is obstacle".
                p = p_max_obstacle
            else:
                # Sensor has finite hit and map has finite hit.
                error = r_eff - r_expected

                p_hit = gaussian_norm * exp(-0.5 * (error / sigma_hit) ** 2)
                p_rand = 1.0 / max_sensor_range

                p = z_hit * p_hit + z_rand * p_rand

                abs_error_sum += abs(error)
                abs_error_count += 1

        else:
            no_map_hit_count += 1

            if measured_no_hit:
                # Sensor says "nothing", map also says "nothing".
                p = p_max_no_obstacle
            else:
                # Sensor saw an obstacle, but map has none along this ray.
                p = p_no_obstacle_for_hit

        if p < eps:
            p = eps

        log_likelihood += log(p)

    mean_abs_error = 0.0
    if abs_error_count > 0:
        mean_abs_error = abs_error_sum / abs_error_count

    return (
        log_likelihood,
        mean_abs_error,
        valid_beam_count,
        map_hit_count,
        no_map_hit_count,
        out_of_map_count,
    )


class BeamRangeFinderModel(MeasurementModel):
    """
    Small Python wrapper around the Numba raytracing likelihood.

    Expected OGM object fields:
        ogm.log_odds_map
        ogm.shift_x
        ogm.shift_y
        ogm.grid_resolution_m
        ogm.min_sensor_range
        ogm.max_sensor_range
    """

    def __init__(
        self,
        occ_thresh: float = 1.4,
        sigma_hit: float = 0.15,
        z_hit: float = 0.95,
        z_rand: float = 0.05,
        p_max_no_obstacle: float = 0.8,
        p_max_obstacle: float = 0.02,
        p_no_obstacle_for_hit: float = 0.01,
        beam_step: int = 2,
    ):
        self.occ_thresh = occ_thresh
        self.sigma_hit = sigma_hit

        self.z_hit = z_hit
        self.z_rand = z_rand

        self.p_max_no_obstacle = p_max_no_obstacle
        self.p_max_obstacle = p_max_obstacle
        self.p_no_obstacle_for_hit = p_no_obstacle_for_hit

        self.beam_step = beam_step


    def likelihood(self, pose: Pose2D, measurements: List[Tuple[float, float]], ogm: OGM) -> dict:
        """
        Parameters
        ----------
        pose : tuple
            (x, y, heading)

        measurements : array-like
            Shape (N, 2), each row = (range, bearing)

        ogm : OGM
            occupancy grid map object.

        Returns
        -------
        dict
            Contains log likelihood and diagnostics.
        """

        measurements_np = np.asarray(measurements, dtype=np.float64)

        if measurements_np.size == 0:
            return {
                "log_likelihood": -1.0e12,
                "mean_abs_error": 0.0,
                "valid_beam_count": 0,
                "map_hit_count": 0,
                "no_map_hit_count": 0,
                "out_of_map_count": 0,
            }

        x, y, heading = pose

        (
            log_likelihood,
            mean_abs_error,
            valid_beam_count,
            map_hit_count,
            no_map_hit_count,
            out_of_map_count,
        ) = raytracing_log_likelihood_numba(
            log_odds_map=ogm.return_log_odds_map(),
            measurements=measurements_np,
            x=x,
            y=y,
            heading=heading,
            shift_x=ogm.shift_x,
            shift_y=ogm.shift_y,
            grid_resolution=ogm.grid_resolution_m,
            min_sensor_range=ogm.min_sensor_range,
            max_sensor_range=ogm.max_sensor_range,
            occ_thresh=self.occ_thresh,
            sigma_hit=self.sigma_hit,
            z_hit=self.z_hit,
            z_rand=self.z_rand,
            p_max_no_obstacle=self.p_max_no_obstacle,
            p_max_obstacle=self.p_max_obstacle,
            p_no_obstacle_for_hit=self.p_no_obstacle_for_hit,
            beam_step=self.beam_step,
        )

        return {
            "log_likelihood": log_likelihood,
            "mean_abs_error": mean_abs_error,
            "valid_beam_count": valid_beam_count,
            "map_hit_count": map_hit_count,
            "no_map_hit_count": no_map_hit_count,
            "out_of_map_count": out_of_map_count,
        }


    # def likelihood(
    #     self,
    #     pose: Pose2D,
    #     measurements: List[Tuple[float, float]],
    #     scan_matcher,
    #     neighbor,
    # ) -> float:
    #     '''
    #     Gmapping version of likelihood computation
    #     '''
        
    #     return float(1.0)

    
    def likelihood_batch(
        self,
        poses: np.ndarray,
        measurements: List[Tuple[float, float]],
        scan_matcher,
        neighbor,
    ) -> np.ndarray:
        '''
        Gmapping variant of lieklihood computation.

        TODO: 
        If used later on we need to make the compuation of the max_distances dependend on grid resolution. 
        max_distance = np.sqrt(2.0) * (kernel_size + 0.5) * resolution
        
        '''
        # Compute no git lieklihood for punishment
        
        return np.full(poses.shape[0], 1e-9)


    def likelihood_batch_copy(
        self,
        poses: np.ndarray,
        measurements: List[Tuple[float, float]],
        scan_matcher,
        neighbor,
    ) -> np.ndarray:
        '''
        Classic NN based computation of measurement likelihood. We use the NN from the scan matcher and compute the
        likelihood for every given pose based on the distance between the measurement endpoints and the nearest neighbor
        in the map points. 
        '''
        return np.full(poses.shape[0], 1e-9)
    

    def gmapping_likelihood(
        self,
        pose: Pose2D,
        measurements: Tuple[float, float],
        ogm: OGM,        
        usable_range: float,
        kernel_size: int=1,
        fullness_threshold: float=1.2,
        free_threshold: float=1.2,
        gaussian_sigma: float=0.05,
        free_cell_ratio: float=np.sqrt(2.0),
    ) -> Tuple[float, float, int]:
        

        return 1.0, 1.0, 1

