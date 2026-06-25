from typing import List, Tuple

import numpy as np
from numba import njit
from math import floor, sqrt, exp, log, pi, erf

from .measurement_model import MeasurementModel
from slam.infrastructure.defs import Pose2D
from slam.scan_matcher.ogm_scan_matching import OGM


# @njit
# def _normal_cdf(x: float) -> float:
#     """
#     Standard normal CDF.
#     Used for truncated Gaussian normalization.
#     """
#     return 0.5 * (1.0 + erf(x / sqrt(2.0)))


@njit(cache=True, nogil=True)
def _beam_model_prob(
    z: float,
    z_exp: float,
    measured_max: bool,
    max_sensor_range: float,
    sigma_hit: float,
    lambda_short: float,
    w_hit: float,
    w_short: float,
    w_max: float,
    w_rand: float,
    eps: float,
) -> float:
    """
    Probabilistic Robotics beam model:

        p(z | x, m) =
            w_hit   * p_hit
          + w_short * p_short
          + w_max   * p_max
          + w_rand  * p_rand

    Important naming:
        max_sensor_range = physical laser max range
        w_max            = mixture weight for max-range component
    """

    # Safety
    if sigma_hit <= 0.0:
        sigma_hit = 1e-6

    if lambda_short <= 0.0:
        lambda_short = 1e-6

    # Clamp z into valid sensor range.
    if z < 0.0:
        z = 0.0

    if z > max_sensor_range:
        z = max_sensor_range

    # ------------------------------------------------------------
    # p_hit: Gaussian around expected measurement, truncated to [0, max_range]
    # ------------------------------------------------------------
    p_hit = 0.0

    if z >= 0.0 and z <= max_sensor_range:
        # Compute normalizer
        gaussian_norm = 1.0 / (sigma_hit * sqrt(2.0 * pi))
        # Compute likelihood for valid z
        p_hit = gaussian_norm * exp(-0.5 * ((z - z_exp) / sigma_hit) ** 2)

    # ------------------------------------------------------------
    # p_short: unexpected obstacle before expected obstacle
    # valid only for 0 <= z < z_exp
    # ------------------------------------------------------------
    p_short = 0.0

    if z >= 0.0 and z < z_exp:
        denom = 1.0 - exp(-lambda_short * z_exp)
        if denom > eps:
            eta_short = 1.0 / denom
            p_short = eta_short * lambda_short * exp(-lambda_short * z)

    # ------------------------------------------------------------
    # p_max: max range / failure component
    # point mass approximated by 1 if measured max, else 0
    # ------------------------------------------------------------
    p_max = 1.0 if measured_max else 0.0

    # ------------------------------------------------------------
    # p_rand: random measurement, uniform over [0, max_range]
    # ------------------------------------------------------------
    p_rand = 0.0
    if z >= 0.0 and z < max_sensor_range:
        p_rand = 1.0 / max_sensor_range

    # Mixture
    prob = (
        w_hit * p_hit
        + w_short * p_short
        + w_max * p_max
        + w_rand * p_rand
    )

    if prob < eps:
        prob = eps

    return prob


@njit(cache=True, nogil=True)
def _raytrace_first_occupied_cell(
    log_odds_map: np.ndarray,
    pose_i: int,
    pose_j: int,
    end_i: int,
    end_j: int,
    occ_thresh: float,
    free_thresh: float,
):
    """
    Bresenham raytracing from robot cell to beam endpoint. The goalis to find the first pccupied cell in a straight line
    between the given pose and the end cell.

    Returns
    -------
    found : bool
        True if occupied cell was found.

    hit_i, hit_j : int
        First occupied cell along ray.

    out_of_map : bool
        True if ray left the map.

    free_count : int
        Number of known-free cells along the ray.

    unknown_count : int
        Number of unknown cells along the ray.

    total_count : int
        Number of evaluated cells along the ray, excluding robot start cell.
    """
    # Init vars
    found = False
    out_of_map = False
    free_count = 0
    unknown_count = 0
    total_count = 0

    n_rows, n_cols = log_odds_map.shape

    # Init cell counter
    cell_i = pose_i
    cell_j = pose_j

    # Compute dx and dy to end cell
    dx = abs(end_j - cell_j)
    dy = abs(end_i - cell_i)

    # Define direction for iteration
    sx = 1 if cell_j < end_j else -1
    sy = 1 if cell_i < end_i else -1

    err = dx - dy
    first_cell = True   

    while True:
        # Check if ray left the map
        if cell_i < 0 or cell_i >= n_rows or cell_j < 0 or cell_j >= n_cols:
            found = False
            out_of_map = True
            return found, -1, -1, out_of_map, free_count, unknown_count, total_count

        # Do not evaluate robot's own cell
        if not first_cell:
            # Extract log odds val
            val = log_odds_map[cell_i, cell_j]
            total_count += 1

            # check if occupied cell has been found
            if val >= occ_thresh:
                found = True
                out_of_map = False
                return found, cell_i, cell_j, out_of_map, free_count, unknown_count, total_count

            # Check if free cell has been found
            elif val <= free_thresh:
                free_count += 1

            # Check if unknown cell has been found (when not occ and not free -> unknown)
            else:
                unknown_count += 1
        
        else:
            first_cell = False

        # Endpoint reached
        if cell_i == end_i and cell_j == end_j:
            break
        
        # Increment counters
        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            cell_j += sx

        if e2 < dx:
            err += dx
            cell_i += sy

    return found, -1, -1, out_of_map, free_count, unknown_count, total_count


@njit(cache=True, nogil=True)
def raytracing_log_likelihood_numba(
    # OGM params
    log_odds_map: np.ndarray,
    shift_x: float,
    shift_y: float,
    occ_thresh: float,
    free_thresh: float,
    
    grid_resolution: float,
    
    min_sensor_range: float,
    max_sensor_range: float,
    measurements: np.ndarray,      # shape (N, 2): [range, bearing]
    
    x: float,
    y: float,
    heading: float,
    
    # Occupancy classification
    unknown_ratio_thresh: float,
    known_free_ratio_thresh: float,

    # Book beam model params
    sigma_hit: float,
    lambda_short: float,
    w_hit: float,
    w_short: float,
    w_max: float,
    w_rand: float,

    # Default probs for special cases
    p_unknown: float,
    p_out_of_map: float,
    p_unexpected_known_free: float,
    p_pred_below_min: float,

    # Numerical / scaling
    alpha_meas: float,
    beam_step: int,
    eps: float,
):
    """
    Beam Range Finder Model based on Probabilistic Robotics,
    extended for occupancy-grid SLAM cases:

    - pose outside map
    - invalid measurements
    - max-range measurements
    - expected map hit
    - known-free no-hit rays
    - unknown rays
    - out-of-map rays

    Returns
    -------
    log_likelihood : float
            The log likelihood of the given pose based on the given measurements and map.
    mean_abs_error : float
        The mean error of all valid errors between measurement range and predicted measurement range
    valid_beam_count : int
        The number of beams after filtering invalid measurements
    map_hit_count : int
        Counter that increments everytime a occupied cell has been found by raytracing.    
    no_map_hit_count : int
        Counter that increments everytime no occupied cell has been found by raytracing.
    out_of_map_count : int
        Counter that increments everytime the raytracing left the map before finding an occupied cell.
    unknown_ray_count : int
        Only computed if the model hasn't found an occupied cell. We count the number of unknown rays everytime a beam traversed
        a cell which occ value is unknown. Then we compute the rate by dividing through the total number of cells that have been 
        traversed. If this is higher than unknown_ratio_thresh then we increase this counter value.
    known_free_ray_count : int
        Same as unknown_ray_count but this time we computing the rate based on the free cells the beam traversed.
        Then we check if this rate is greater than known_free_ratio_thresh.
    unexpected_known_free_count : int
        Counter increments everytime the raytracing havent detected an occ cell and the actual measruement is not max range measurement.
        In this the measruement detected an unexpected obstacle while the number of free and unknown cell haven been above threshold.
    skipped_beam_count : int
        Counter that increases everytime we skip a beam due to invalid measurement (nan, below min range, etc.)
    """
    # Init
    # Comptued log likelihood 
    log_likelihood = 0.0

    # The mean error of all valid errors between measurement range and predicted measurement range
    mean_abs_error = 0.0
    
    valid_beam_count = 0
    
    map_hit_count = 0
    no_map_hit_count = 0
    out_of_map_count = 0
    
    unknown_ray_count = 0
    known_free_ray_count = 0
    unexpected_known_free_count = 0    
    
    skipped_beam_count = 0

    abs_error_sum = 0.0
    abs_error_count = 0
    

    # Transform given position into grid cell and check for valid cell
    n_rows, n_cols = log_odds_map.shape

    pose_i = int(floor((y + shift_y) / grid_resolution))
    pose_j = int(floor((x + shift_x) / grid_resolution))

    # Check if robot position out of map -> punish 
    if pose_i < 0 or pose_i >= n_rows or pose_j < 0 or pose_j >= n_cols:
        log_likelihood = -1.0e12
        return (
            log_likelihood,
            mean_abs_error,
            valid_beam_count,
            map_hit_count,
            no_map_hit_count,
            out_of_map_count,
            unknown_ray_count,
            known_free_ray_count,
            unexpected_known_free_count,
            skipped_beam_count,
        )

    # Ensure valid beam step
    if beam_step < 1:
        beam_step = 1

    # Check if max sensor range is valid
    if max_sensor_range <= min_sensor_range:
        log_likelihood = -1.0e12
        return (
            log_likelihood,
            mean_abs_error,
            valid_beam_count,
            map_hit_count,
            no_map_hit_count,
            out_of_map_count,
            unknown_ray_count,
            known_free_ray_count,
            unexpected_known_free_count,
            skipped_beam_count,
        )

    # Set default scale value for log likelihood if given one is not valid
    if alpha_meas <= 0.0:
        alpha_meas = 1.0

    # Normalize mixture weights
    w_sum = w_hit + w_short + w_max + w_rand
    if w_sum <= eps:
        w_hit = 0.70
        w_short = 0.10
        w_max = 0.10
        w_rand = 0.10
        w_sum = 1.0

    w_hit /= w_sum
    w_short /= w_sum
    w_max /= w_sum
    w_rand /= w_sum
    
    # Compute likelihood for every beam
    for k in range(0, measurements.shape[0], beam_step):
        r_meas = measurements[k, 0]
        bearing = measurements[k, 1]

        # Skip invalid measurements
        # Skip nan measurements
        if np.isnan(r_meas):
            skipped_beam_count += 1
            continue

        # Skip measurements below minimum range
        if r_meas <= min_sensor_range:
            skipped_beam_count += 1
            continue
        
        valid_beam_count += 1

        # Ensure valid max range measurements
        measured_max = False

        # Handle +inf and ranges above max range as max range
        if (not np.isfinite(r_meas)) or r_meas >= max_sensor_range:
            measured_max = True
            z = max_sensor_range
        else:
            z = r_meas

        # Estimate raytracing position of beam by max sensor range raycasting.
        # Compute reflecting grid cell for max measurement along beam direction
        phi = heading + bearing

        ray_end_x = x + max_sensor_range * np.cos(phi)
        ray_end_y = y + max_sensor_range * np.sin(phi)

        end_i = int(floor((ray_end_y + shift_y) / grid_resolution))
        end_j = int(floor((ray_end_x + shift_x) / grid_resolution))

        (
            found,
            hit_i,
            hit_j,
            out_of_map,
            free_count,
            unknown_count,
            total_count,
        ) = _raytrace_first_occupied_cell(
            log_odds_map=log_odds_map,
            pose_i=pose_i,
            pose_j=pose_j,
            end_i=end_i,
            end_j=end_j,
            occ_thresh=occ_thresh,
            free_thresh=free_thresh,
        )

        # Case 1: If the ray left the map before occupied cell has been found -> punish
        if out_of_map:
            out_of_map_count += 1
            prob = p_out_of_map

        # Case 2: If we found the occupied cell in map -> increase hit count and compute likelihood with beam model
        elif found:
            map_hit_count += 1

            # Transform reflecting cell to expected range
            hit_x = hit_j * grid_resolution - shift_x + grid_resolution / 2.0
            hit_y = hit_i * grid_resolution - shift_y + grid_resolution / 2.0

            dx = hit_x - x
            dy = hit_y - y
            z_exp = sqrt(dx * dx + dy * dy)

            # Validate that predicted range is above threshold
            if z_exp <= min_sensor_range:
                prob = p_pred_below_min

            else:
                # Compute measurement likelihood with Laser Range Finder Model
                prob = _beam_model_prob(
                    z=z,
                    z_exp=z_exp,
                    measured_max=measured_max,
                    max_sensor_range=max_sensor_range,
                    sigma_hit=sigma_hit,
                    lambda_short=lambda_short,
                    w_hit=w_hit,
                    w_short=w_short,
                    w_max=w_max,
                    w_rand=w_rand,
                    eps=eps,
                )

                if not measured_max:
                    abs_error_sum += abs(z - z_exp)
                    abs_error_count += 1

        # Case 3: We have done raytracing, stayed inside the map but haven't found an occupied cell
        else:
            # Increment no hit counter value since we haven't found an occupied cell
            no_map_hit_count += 1
            
            # Set unknown ratio and free ratio
            unknown_ratio = 1.0
            free_ratio = 0.0

            # If we have traversed cells then compute unknown and free ratios
            if total_count > 0:
                unknown_ratio = unknown_count / total_count
                free_ratio = free_count / total_count

            # If the we raytraced over more than unknown_ratio_thresh unknown cells, then we use small punishment
            if unknown_ratio >= unknown_ratio_thresh:
                unknown_ray_count += 1
                prob = p_unknown
            
            # If the free ratio is higher than known_free_ratio_thresh, then we compute likelihood with beam model
            elif free_ratio >= known_free_ratio_thresh:
                known_free_ray_count += 1
                z_exp = max_sensor_range

                prob = _beam_model_prob(
                    z=z,
                    z_exp=z_exp,
                    measured_max=measured_max,
                    max_sensor_range=max_sensor_range,
                    sigma_hit=sigma_hit,
                    lambda_short=lambda_short,
                    w_hit=w_hit,
                    w_short=w_short,
                    w_max=w_max,
                    w_rand=w_rand,
                    eps=eps,
                )

                # If sensor measurement is not max range measruement and expected measurement is max range, than increase unexpected 
                # known free count
                if not measured_max:
                    unexpected_known_free_count += 1

                    # If the computed prbo is below the given threshold, then we set it to the given threshold value. 
                    if prob < p_unexpected_known_free:
                        prob = p_unexpected_known_free

            # ----------------------------------------------------
            # Mixed free/unknown ray -> map prediction is weak
            # ----------------------------------------------------
            else:
                unknown_ray_count += 1
                prob = p_unknown

        # Ensure valid likelihood
        if prob < eps:
            prob = eps

        # Transform to log space and accumulate log likelihoods
        log_likelihood += log(prob)

    # No valid beams -> neutral update, not particle death.
    # This means: measurement gives no information.
    if valid_beam_count <= 0:
        # log_likelihood = 0.0 -> prob = 1.0 -> neutral update, doesn't change the weight
        log_likelihood = 0.0
        return (
            log_likelihood,
            mean_abs_error,
            valid_beam_count,
            map_hit_count,
            no_map_hit_count,
            out_of_map_count,
            unknown_ray_count,
            known_free_ray_count,
            unexpected_known_free_count,
            skipped_beam_count,
        )
        
    # Compute mean absolute error
    mean_abs_error = 0.0
    if abs_error_count > 0:
        mean_abs_error = abs_error_sum / abs_error_count

    # Weight estimated log likelihoods for certain cases (combination of measurement + motion probs, etc.)
    log_likelihood *= alpha_meas

    return (
        log_likelihood,
        mean_abs_error,
        valid_beam_count,
        map_hit_count,
        no_map_hit_count,
        out_of_map_count,
        unknown_ray_count,
        known_free_ray_count,
        unexpected_known_free_count,
        skipped_beam_count,
    )


class BeamRangeFinderModel(MeasurementModel):
    """
    Probabilistic Robotics Beam Range Finder Model for occupancy-grid SLAM.

    Uses the book model:

        p(z | x, m) =
            w_hit   * p_hit
          + w_short * p_short
          + w_max   * p_max
          + w_rand  * p_rand

    plus extra handling for occupancy-grid-specific cases:

        - pose outside map
        - unknown map rays
        - out-of-map rays
        - known-free no-hit rays
        - too-close/invalid beams

    Important:
        max_sensor_range is the physical laser range.
        w_max is the mixture weight for the max-range component.
    """

    def __init__(
        self,

        # Occupancy classification
        occ_thresh: float = 1.4,
        free_thresh: float = -1.4,
        unknown_ratio_thresh: float = 0.30,
        known_free_ratio_thresh: float = 0.70,

        # Book beam model parameters
        sigma_hit: float = 0.15,
        w_hit: float = 0.70,      
        w_short: float = 0.10,
        lambda_short: float = 0.20,  
        w_max: float = 0.10,
        w_rand: float = 0.10,

        # Default probs for special cases
        p_unknown: float = 0.20,
        p_out_of_map: float = 0.10,
        p_unexpected_known_free: float = 0.00,
        p_pred_below_min: float = 0.02,

        # Numerical / scaling
        alpha_meas: float = 0.10,
        beam_step: int = 2,
        eps: float = 1e-12,
    ):
        self.occ_thresh = occ_thresh
        self.free_thresh = free_thresh
        self.unknown_ratio_thresh = unknown_ratio_thresh
        self.known_free_ratio_thresh = known_free_ratio_thresh

        self.sigma_hit = sigma_hit
        self.lambda_short = lambda_short

        self.w_hit = w_hit
        self.w_short = w_short
        self.w_max = w_max
        self.w_rand = w_rand

        self.p_unknown = p_unknown
        self.p_out_of_map = p_out_of_map
        self.p_unexpected_known_free = p_unexpected_known_free
        self.p_pred_below_min = p_pred_below_min

        self.alpha_meas = alpha_meas
        self.beam_step = beam_step
        self.eps = eps


    def likelihood(
        self,
        pose: Pose2D,
        measurements: List[Tuple[float, float]],
        ogm: OGM,
    ) -> dict:
        """
        Get's a 2D pose, the range, bearing measruements and an ogm object to compute the likelihood 
        for the robot being in teh given pose, based on teh given data. 
        The method filters nan values from the measruements. If u want them to count as max range measurements, then 
        set all nan to max range before.

        Parameters
        ----------
        pose:
            (x, y, theta)

        measurements:
            List/array of (range, bearing)

        ogm:
            Occupancy grid map.

        Returns
        -------
        dict with:
            log_likelihood
            mean_abs_error
            valid_beam_count
            map_hit_count
            no_map_hit_count
            out_of_map_count
            unknown_ray_count
            known_free_ray_count
            unexpected_known_free_count
            skipped_beam_count
        """

        measurements_np = np.asarray(measurements, dtype=np.float64)

        # Empty scan -> neutral update.
        # No measurement information means no weight change.
        if measurements_np.size == 0:
            return {
                "log_likelihood": 0.0,
                "mean_abs_error": 0.0,
                "valid_beam_count": 0,
                "map_hit_count": 0,
                "no_map_hit_count": 0,
                "out_of_map_count": 0,
                "unknown_ray_count": 0,
                "known_free_ray_count": 0,
                "unexpected_known_free_count": 0,
                "skipped_beam_count": 0,
            }

        # Extract robot pose
        x, y, heading = pose

        # Compute measurement likelihood 
        (
            log_likelihood,
            mean_abs_error,
            valid_beam_count,
            map_hit_count,
            no_map_hit_count,
            out_of_map_count,
            unknown_ray_count,
            known_free_ray_count,
            unexpected_known_free_count,
            skipped_beam_count,
        ) = raytracing_log_likelihood_numba(
            log_odds_map=ogm.return_log_odds_map(),
            
            shift_x=ogm.shift_x,
            shift_y=ogm.shift_y,
            occ_thresh=self.occ_thresh,
            free_thresh=self.free_thresh,
            grid_resolution=ogm.grid_resolution_m,
            
            min_sensor_range=ogm.min_sensor_range,
            max_sensor_range=ogm.max_sensor_range,
            measurements=measurements_np,

            x=x,
            y=y,
            heading=heading,
            
            unknown_ratio_thresh=self.unknown_ratio_thresh,
            known_free_ratio_thresh=self.known_free_ratio_thresh,

            sigma_hit=self.sigma_hit,
            lambda_short=self.lambda_short,
            w_hit=self.w_hit,
            w_short=self.w_short,
            w_max=self.w_max,
            w_rand=self.w_rand,

            p_unknown=self.p_unknown,
            p_out_of_map=self.p_out_of_map,
            p_unexpected_known_free=self.p_unexpected_known_free,
            p_pred_below_min=self.p_pred_below_min,

            alpha_meas=self.alpha_meas,
            beam_step=self.beam_step,
            eps=self.eps,
        )
      
        return {
            "log_likelihood": float(log_likelihood),
            "mean_abs_error": float(mean_abs_error),

            "valid_beam_count": int(valid_beam_count),
            "map_hit_count": int(map_hit_count),
            "no_map_hit_count": int(no_map_hit_count),
            "out_of_map_count": int(out_of_map_count),
            "unknown_ray_count": int(unknown_ray_count),
            "known_free_ray_count": int(known_free_ray_count),
            "unexpected_known_free_count": int(unexpected_known_free_count),
            "skipped_beam_count": int(skipped_beam_count),
        }


    def likelihood_batch(
        self,
        poses: np.ndarray,
        measurements: List[Tuple[float, float]],
        scan_matcher,
        neighbor=None,
    ) -> np.ndarray:
        """
        Compatibility helper.

        Returns log likelihoods for all poses.
        This is not used in your current range-finder proposal path,
        but keeping it avoids interface surprises.
        """

        values = np.empty(poses.shape[0], dtype=np.float64)

        for i in range(poses.shape[0]):
            result = self.likelihood(
                pose=(poses[i, 0], poses[i, 1], poses[i, 2]),
                measurements=measurements,
                ogm=scan_matcher.ogm,
            )
            values[i] = result["log_likelihood"]

        return values


    def likelihood_batch_copy(
        self,
        poses: np.ndarray,
        measurements: List[Tuple[float, float]],
        scan_matcher,
        neighbor=None,
    ) -> np.ndarray:
        """
        Compatibility wrapper.
        """
        return self.likelihood_batch(
            poses=poses,
            measurements=measurements,
            scan_matcher=scan_matcher,
            neighbor=neighbor,
        )


    def gmapping_likelihood(
        self,
        pose: Pose2D,
        measurements: Tuple[float, float],
        ogm: OGM,
        usable_range: float,
        kernel_size: int = 1,
        fullness_threshold: float = 1.2,
        free_threshold: float = 1.2,
        gaussian_sigma: float = 0.05,
        free_cell_ratio: float = np.sqrt(2.0),
    ) -> Tuple[float, float, int]:
        """
        Compatibility stub for older code paths.

        Returns
        -------
        score, log_likelihood, matched_count
        """

        result = self.likelihood(
            pose=pose,
            measurements=measurements,
            ogm=ogm,
        )

        log_likelihood = result["log_likelihood"]
        matched_count = result["valid_beam_count"]

        # Score is not used meaningfully here; return exp only safely for small values.
        score = 0.0
        if log_likelihood > -700.0:
            score = float(np.exp(log_likelihood))

        return score, log_likelihood, matched_count