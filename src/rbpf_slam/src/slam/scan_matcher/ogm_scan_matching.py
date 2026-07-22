#!/usr/bin/env python3

import rospy
from typing import List, Tuple, Optional
import threading

from numba import njit
import numpy as np
from math import exp, atan2, sin, cos, radians, degrees, floor, ceil, isfinite, log
import time
from geometry_msgs.msg import Pose, Point
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
# from nav_msgs.msg import OccupancyGrid
# from rvc_commander.msg import Measurement
# from rvc_commander.msg import LogOddsMap
from rbpf_slam.msg import Measurement
from rbpf_slam.msg import LogOddsMap


# @njit
# def update_map_optimized(
#     log_odds_map,
#     measurements,
#     x,
#     y,
#     heading,
#     shift_x,
#     shift_y,
#     grid_resolution,
#     min_sensor_range,
#     max_sensor_range,
#     log_odds_decreasing,
#     log_odds_increasing,
#     min_log_odds,
#     max_log_odds,
# ):
#     """
#     Update the occupancy grid map (log-odds) using laser scan measurements.

#     This implementation enforces the rule:
#         "Each grid cell is updated at most once per scan"

#     and resolves conflicts using:
#         "Occupied (hit) dominates free (pass-through)"

#     This avoids contradictory updates when multiple beams:
#         - pass through a cell (free update)
#         - and/or terminate in the same cell (occupied update)


    
#     Key Idea
    
#     Instead of updating the map directly for each beam, we:

#     1. Collect all cells touched by beams during this scan
#        → store (i, j, type)
#          type = 0 → free (beam passed)
#          type = 1 → occupied (beam endpoint)

#     2. After processing all beams:
#        → resolve conflicts per cell:
#             if ANY beam marked cell as occupied → occupied update
#             else → free update

#     This ensures:
#         - no double updates per scan
#         - no cancellation (free vs occupied)
#         - stable map behavior

#     -------------------------------------------------------------------------
#     Performance Notes
#     -------------------------------------------------------------------------
#     - Uses fixed-size arrays → Numba compatible
#     - Avoids Python sets/dicts
#     - Two-pass approach:
#         pass 1 → collect cells
#         pass 2 → resolve + update
#     -------------------------------------------------------------------------
    
#     Parameters
#     ---------
    
#     log_odds_map : np.ndarray (H x W)
#         2D occupancy grid in log-odds representation (modified in-place)

#     measurements : np.ndarray (N x 2)
#         Laser measurements: (range, bearing)

#     x, y, heading : float
#         Robot pose in world coordinates

#     shift_x, shift_y : float
#         Map origin shift (map centered around (0,0))

#     grid_resolution : float
#         Size of one grid cell (meters)

#     min_sensor_range, max_sensor_range : float
#         Sensor limits

#     log_odds_decreasing : float
#         Update value for free cells (negative)

#     log_odds_increasing : float
#         Update value for occupied cells (positive)

#     min_log_odds, max_log_odds : float
#         Clamping bounds


#     Returns
#     -------
#     status : int
#         0 = success
#         1 = robot outside map

#     beam_out_map_count : int
#         (not used here, always 0 for now)

#     """

#     n_rows, n_cols = log_odds_map.shape

#     # Convert robot pose to grid indices
#     pose_i = int((y + shift_y) / grid_resolution)
#     pose_j = int((x + shift_x) / grid_resolution)

#     # Abort if robot is outside map
#     if pose_i < 0 or pose_i >= n_rows or pose_j < 0 or pose_j >= n_cols:
#         return 1, 0

#     # ------------------------------------------------------------------
#     # TEMP STORAGE FOR THIS SCAN
#     # ------------------------------------------------------------------
#     # Stores all visited cells (i, j) and their type:
#     #   type = 0 → free cell
#     #   type = 1 → occupied cell (endpoint)
#     #
#     # NOTE:
#     # max_cells must be large enough to hold all traversed cells
#     # Typical beams ~720 → each ray ~100 cells → safe upper bound
#     max_cells = 10000

#     visited_i = np.empty(max_cells, dtype=np.int32)
#     visited_j = np.empty(max_cells, dtype=np.int32)
#     visited_type = np.empty(max_cells, dtype=np.int8)

#     visited_count = 0

#     # ------------------------------------------------------------------
#     # PASS 1: COLLECT ALL CELLS (NO MAP UPDATE)
#     # ------------------------------------------------------------------
#     for k in range(measurements.shape[0]):

#         r = measurements[k, 0]
#         bearing = measurements[k, 1]

#         # Ignore too close measurements
#         if r <= min_sensor_range:
#             continue

#         # Determine measurement type
#         if not np.isfinite(r) or r >= max_sensor_range:
#             r_eff = max_sensor_range
#             is_hit = False   # no endpoint
#         else:
#             r_eff = r
#             is_hit = True    # valid hit

#         # Compute endpoint in world coordinates
#         phi = heading + bearing

#         end_x = x + r_eff * np.cos(phi)
#         end_y = y + r_eff * np.sin(phi)

#         # Convert endpoint to grid indices
#         end_i = int((end_y + shift_y) / grid_resolution)
#         end_j = int((end_x + shift_x) / grid_resolution)

#         # Skip if endpoint outside map
#         if end_i < 0 or end_i >= n_rows or end_j < 0 or end_j >= n_cols:
#             continue

#         # ------------------------------------------------------------------
#         # Bresenham ray tracing
#         # ------------------------------------------------------------------
#         cell_i = pose_i
#         cell_j = pose_j

#         dx = abs(end_j - cell_j)
#         dy = abs(end_i - cell_i)

#         sx = 1 if cell_j < end_j else -1
#         sy = 1 if cell_i < end_i else -1

#         err = dx - dy

#         while True:

#             # ----------------------------------------------------------
#             # STORE CELL IN TEMP BUFFER
#             # ----------------------------------------------------------
#             if visited_count < max_cells:

#                 # Endpoint
#                 if cell_i == end_i and cell_j == end_j:
#                     if is_hit:
#                         visited_i[visited_count] = cell_i
#                         visited_j[visited_count] = cell_j
#                         visited_type[visited_count] = 1  # occupied
#                         visited_count += 1

#                 # Free cell
#                 else:
#                     visited_i[visited_count] = cell_i
#                     visited_j[visited_count] = cell_j
#                     visited_type[visited_count] = 0  # free
#                     visited_count += 1

#             # Stop at endpoint
#             if cell_i == end_i and cell_j == end_j:
#                 break

#             # Bresenham step
#             e2 = 2 * err
#             if e2 > -dy:
#                 err -= dy
#                 cell_j += sx
#             if e2 < dx:
#                 err += dx
#                 cell_i += sy

#             # Safety bounds check
#             if cell_i < 0 or cell_i >= n_rows or cell_j < 0 or cell_j >= n_cols:
#                 break

#     # ------------------------------------------------------------------
#     # PASS 2: RESOLVE CONFLICTS + UPDATE MAP
#     # ------------------------------------------------------------------
#     for idx in range(visited_count):

#         i = visited_i[idx]
#         j = visited_j[idx]

#         # Check if ANY beam marked this cell as occupied
#         is_occupied = False

#         for k in range(visited_count):
#             if visited_i[k] == i and visited_j[k] == j:
#                 if visited_type[k] == 1:
#                     is_occupied = True
#                     break

#         old_val = log_odds_map[i, j]

#         # Skip if already saturated
#         if old_val <= min_log_odds or old_val >= max_log_odds:
#             continue

#         # Apply update
#         if is_occupied:
#             log_odds_map[i, j] = old_val + log_odds_increasing
#         else:
#             log_odds_map[i, j] = old_val + log_odds_decreasing

#     return 0, 0



@njit(cache=True, nogil=True)
def update_map_numba_unique_cells(
    log_odds_map,
    measurements,
    x,
    y,
    heading,
    shift_x,
    shift_y,
    grid_resolution,
    min_sensor_range,
    max_sensor_range,
    log_odds_decreasing,
    log_odds_increasing,
    min_log_odds,
    max_log_odds,
):
    """
    Update log-odds occupancy grid using one update per cell per scan.

    Cell states for this scan:
        0 = untouched
        1 = free candidate
        2 = occupied endpoint candidate

    Rule:
        - If at least one beam ends in a cell, the cell gets one occupied update.
        - Else if one or more beams pass through a cell, the cell gets one free update.
        - A cell is never updated more than once per scan.
        - Occupied endpoints dominate free pass-through evidence.
    """

    n_rows, n_cols = log_odds_map.shape

    pose_i = int(np.floor((y + shift_y) / grid_resolution))
    pose_j = int(np.floor((x + shift_x) / grid_resolution))

    if pose_i < 0 or pose_i >= n_rows or pose_j < 0 or pose_j >= n_cols:
        return 1, 0

    # 0 untouched, 1 free, 2 occupied
    cell_state = np.zeros((n_rows, n_cols), dtype=np.uint8)

    beam_out_map_count = 0

    # ------------------------------------------------------------
    # PASS 1: collect all real occupied endpoint cells
    # ------------------------------------------------------------
    for k in range(measurements.shape[0]):
        r = measurements[k, 0]
        bearing = measurements[k, 1]

        if r <= min_sensor_range:
            continue

        # Only finite non-max-range beams create occupied endpoints
        if (not np.isfinite(r)) or r >= max_sensor_range:
            continue

        phi = heading + bearing

        end_x = x + r * np.cos(phi)
        end_y = y + r * np.sin(phi)

        end_i = int(np.floor((end_y + shift_y) / grid_resolution))
        end_j = int(np.floor((end_x + shift_x) / grid_resolution))

        if end_i < 0 or end_i >= n_rows or end_j < 0 or end_j >= n_cols:
            beam_out_map_count += 1
            continue

        cell_state[end_i, end_j] = 2

    # ------------------------------------------------------------
    # PASS 2: trace rays and collect free cells
    #         but never overwrite occupied endpoint cells
    # ------------------------------------------------------------
    for k in range(measurements.shape[0]):
        r = measurements[k, 0]
        bearing = measurements[k, 1]

        if r <= min_sensor_range:
            continue

        if (not np.isfinite(r)) or r >= max_sensor_range:
            r_eff = max_sensor_range
            is_hit = False
        else:
            r_eff = r
            is_hit = True

        phi = heading + bearing

        end_x = x + r_eff * np.cos(phi)
        end_y = y + r_eff * np.sin(phi)

        end_i = int(np.floor((end_y + shift_y) / grid_resolution))
        end_j = int(np.floor((end_x + shift_x) / grid_resolution))

        if end_i < 0 or end_i >= n_rows or end_j < 0 or end_j >= n_cols:
            beam_out_map_count += 1
            continue

        cell_i = pose_i
        cell_j = pose_j

        dx = abs(end_j - cell_j)
        dy = abs(end_i - cell_i)

        sx = 1 if cell_j < end_j else -1
        sy = 1 if cell_i < end_i else -1

        err = dx - dy

        while True:
            is_endpoint = (cell_i == end_i and cell_j == end_j)

            # Do not mark the endpoint of a real hit as free.
            # Also do not overwrite any occupied endpoint cell from any beam.
            if not is_endpoint:
                if cell_state[cell_i, cell_j] != 2:
                    cell_state[cell_i, cell_j] = 1

            # For max-range/no-hit beams, endpoint is also free unless occupied by another beam.
            if is_endpoint:
                if not is_hit:
                    if cell_state[cell_i, cell_j] != 2:
                        cell_state[cell_i, cell_j] = 1
                break

            e2 = 2 * err

            if e2 > -dy:
                err -= dy
                cell_j += sx

            if e2 < dx:
                err += dx
                cell_i += sy

            if cell_i < 0 or cell_i >= n_rows or cell_j < 0 or cell_j >= n_cols:
                break

    # ------------------------------------------------------------
    # PASS 3: apply exactly one update per touched cell
    # ------------------------------------------------------------
    for i in range(n_rows):
        for j in range(n_cols):
            state = cell_state[i, j]

            if state == 0:
                continue

            old_val = log_odds_map[i, j]

            if old_val <= min_log_odds or old_val >= max_log_odds:
                continue

            if state == 2:
                new_val = old_val + log_odds_increasing
            else:
                new_val = old_val + log_odds_decreasing

            if new_val < min_log_odds:
                new_val = min_log_odds
            elif new_val > max_log_odds:
                new_val = max_log_odds

            log_odds_map[i, j] = new_val

    return 0, beam_out_map_count


@njit(cache=True, nogil=True)
def update_map_numba_inf_free_space(
    log_odds_map: np.ndarray,
    measurements: np.ndarray,   # shape (N, 2) -> [range, bearing]
    x: float,
    y: float,
    heading: float,
    shift_x: float,
    shift_y: float,
    grid_resolution: float,
    min_sensor_range: float,
    max_sensor_range: float,
    log_odds_decreasing: float,
    log_odds_increasing: float,
    min_log_odds: float,
    max_log_odds: float,
) -> int:
    """
    Update the occupancy grid map (log-odds) using laser scan measurements.

    This version correctly handles:
    - finite measurements → free space + occupied endpoint
    - max-range / inf measurements → free space ONLY (no occupied endpoint)

    Parameters
    ----------
    log_odds_map : np.ndarray
        2D log-odds occupancy grid map (modified in-place)

    measurements : np.ndarray
        Array of shape (N, 2) containing (range, bearing)

    x, y, heading : float
        Robot pose in world coordinates

    shift_x, shift_y : float
        Map origin shift (centered map)

    grid_resolution : float
        Cell size in meters

    min_sensor_range, max_sensor_range : float
        Sensor limits

    log_odds_decreasing : float
        Log-odds update for free cells (negative)

    log_odds_increasing : float
        Log-odds update for occupied cells (positive)

    min_log_odds, max_log_odds : float
        Clamping limits

    Returns
    -------
    int
        Status flag (0 = success, 1 = robot outside map)
    """
    beam_out_map_count = 0
    n_rows, n_cols = log_odds_map.shape

    # Convert robot pose to grid index
    pose_i = int(np.floor((y + shift_y) / grid_resolution))
    pose_j = int(np.floor((x + shift_x) / grid_resolution))

    # If robot is outside map → abort
    if pose_i < 0 or pose_i >= n_rows or pose_j < 0 or pose_j >= n_cols:
        return 1, beam_out_map_count

    # Iterate over all beams
    for k in range(measurements.shape[0]):
        r = measurements[k, 0]
        bearing = measurements[k, 1]

        # --------------------------------------------------
        # 1. Determine measurement type
        # --------------------------------------------------

        # Ignore too small values
        if r <= min_sensor_range:
            continue

        # Handle max-range / inf → free space only
        if not np.isfinite(r) or r >= max_sensor_range:
            r_eff = max_sensor_range
            is_hit = False   # NO occupied endpoint
        else:
            r_eff = r
            is_hit = True    # valid obstacle hit

        # --------------------------------------------------
        # 2. Compute endpoint in world coordinates
        # --------------------------------------------------
        phi = heading + bearing

        end_x = x + r_eff * np.cos(phi)
        end_y = y + r_eff * np.sin(phi)

        end_i = int(np.floor((end_y + shift_y) / grid_resolution))
        end_j = int(np.floor((end_x + shift_x) / grid_resolution))

        # If endpoint outside map → skip beam
        if end_i < 0 or end_i >= n_rows or end_j < 0 or end_j >= n_cols:
            beam_out_map_count += 1
            continue

        # --------------------------------------------------
        # 3. Bresenham ray tracing (free + occupied update)
        # --------------------------------------------------
        cell_i = pose_i
        cell_j = pose_j

        dx = abs(end_j - cell_j)
        dy = abs(end_i - cell_i)

        sx = 1 if cell_j < end_j else -1
        sy = 1 if cell_i < end_i else -1

        err = dx - dy

        while True:
            old_val = log_odds_map[cell_i, cell_j]

            # Only update if within bounds
            if min_log_odds < old_val < max_log_odds:

                # Last cell (endpoint)
                if cell_i == end_i and cell_j == end_j:
                    if is_hit:
                        # Only mark occupied if real hit
                        log_odds_map[cell_i, cell_j] = old_val + log_odds_increasing
                    # else: max-range → DO NOTHING (no occupied cell)

                else:
                    # Free space update
                    log_odds_map[cell_i, cell_j] = old_val + log_odds_decreasing

            # Stop at endpoint
            if cell_i == end_i and cell_j == end_j:
                break

            # Bresenham step
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cell_j += sx
            if e2 < dx:
                err += dx
                cell_i += sy

            # Safety check
            if cell_i < 0 or cell_i >= n_rows or cell_j < 0 or cell_j >= n_cols:
                break

    return 0, beam_out_map_count


@njit(cache=True, nogil=True)
def extract_map_numba(
    log_odds_map: np.ndarray,
    i_pose: int,
    j_pose: int,
    r_cells: int,
    r_cells_sq: int,
    surface_r_cells: int,
    occ_thresh: float,
    free_thresh: float,
    min_free_count: int,
    grid_res: float,
    shift_x: float,
    shift_y: float
) -> np.ndarray:
    '''
    Map extractor that extracts map points in a circular area around the given pose. To be a valid map point a cell has 
    to fulfill the following criteria:
        1. The cell has to be occupied (log odds value above occ_thresh)    
        2. The cell has to belong to a surface, which is defined as having at least min_free_count free cells 
    
    Parameters
    ----------
    log_odds_map : np.ndarray
        2D log-odds occupancy grid map
    i_pose, j_pose : int
        Robot pose in grid indices
    r_cells : int
        Radius of the circular area in cells
    r_cells_sq : int
        Radius squared (pre-computed for efficiency)
    surface_r_cells : int
        Radius of the surface window in cells (used to check if cell belongs to surface)
    occ_thresh : float
        Log-odds threshold to consider a cell occupied
    free_thresh : float
        Log-odds threshold to consider a cell free
    min_free_count : int
        Minimum number of free cells in the surface window to consider a cell as belonging to a surface
    grid_res : float
        Grid resolution in meters
    shift_x, shift_y : float
        Map origin shift (centered map)
    '''

    n_rows, n_cols = log_odds_map.shape

    # Define maximum number of points that can be extracted and pre-allocate array for points
    # This is a simplistic upper bound cause we use a squared area instead of circular area.
    max_points = (r_cells * 2 + 1) * (r_cells * 2 + 1)
    # points = np.empty((max_points, 2), dtype=np.float64)
    map_points = np.full((max_points, 2), np.nan, dtype=np.float64)
    count = 0

    # With the for loops we define a general square with center cell and it has the size of the radius*2 + 1 
    for di in range(-r_cells, r_cells + 1):
        for dj in range(-r_cells, r_cells + 1):
            
            # For every cell in the general square we check if the cell is inside the radius (from the center point)
            # Using squared values to avoid sqrt -> faster
            # Skip if not inside circle area
            if di*di + dj*dj > r_cells_sq:
                continue
            
            # Compute the actual cell indices in the map/array from our general square and the center point
            i = i_pose + di
            j = j_pose + dj

            # Check if surface window is inside map
            if (
                i < surface_r_cells
                or i >= n_rows - surface_r_cells
                or j < surface_r_cells
                or j >= n_cols - surface_r_cells
            ):
                continue
            
            # Check if cell is free -> skip
            if log_odds_map[i, j] < occ_thresh:
                continue

            # Check if cell belongs to surface
            free_count = 0
            for ni in range(i-surface_r_cells, i+surface_r_cells + 1):
                for nj in range(j-surface_r_cells, j+surface_r_cells + 1):
                    # Exclude pose
                    if ni == i and nj == j:
                        continue
                    # Count number of free cells around map point
                    if log_odds_map[ni, nj] < free_thresh:
                        free_count += 1
            
            # Cell belongs to surface when number of free cells >= min_free_count 
            if free_count < min_free_count:
                continue
            
            # Transform cell indices to point coordinates add to list of valid points
            x = j * grid_res - shift_x + grid_res/2
            y = i * grid_res - shift_y + grid_res/2

            map_points[count, 0] = x
            map_points[count, 1] = y
            count += 1

    return map_points[:count]


@njit(cache=True, nogil=True)
def update_map_numba(
    log_odds_map: np.ndarray,
    measurements: np.ndarray,   # shape (N, 2) -> [range, bearing]
    x: float,
    y: float,
    heading: float,
    shift_x: float,
    shift_y: float,
    grid_resolution: float,
    min_sensor_range: float,
    max_sensor_range: float,
    log_odds_decreasing: float,
    log_odds_increasing: float,
    min_log_odds: float,
    max_log_odds: float,
) -> int:
    """
    Update the occupancy grid map (log-odds) using laser scan measurements.

    This version correctly handles:
    - finite measurements → free space + occupied endpoint
    - max-range / inf measurements → free space ONLY (no occupied endpoint)

    Parameters
    ----------
    log_odds_map : np.ndarray
        2D log-odds occupancy grid map (modified in-place)

    measurements : np.ndarray
        Array of shape (N, 2) containing (range, bearing)

    x, y, heading : float
        Robot pose in world coordinates

    shift_x, shift_y : float
        Map origin shift (centered map)

    grid_resolution : float
        Cell size in meters

    min_sensor_range, max_sensor_range : float
        Sensor limits

    log_odds_decreasing : float
        Log-odds update for free cells (negative)

    log_odds_increasing : float
        Log-odds update for occupied cells (positive)

    min_log_odds, max_log_odds : float
        Clamping limits

    Returns
    -------
    int
        Status flag (0 = success, 1 = robot outside map)
    """
    beam_out_map_count = 0
    n_rows, n_cols = log_odds_map.shape

    # Convert robot pose to grid index
    pose_i = int(np.floor((y + shift_y) / grid_resolution))
    pose_j = int(np.floor((x + shift_x) / grid_resolution))

    # If robot is outside map → abort
    if pose_i < 0 or pose_i >= n_rows or pose_j < 0 or pose_j >= n_cols:
        return 1, beam_out_map_count

    # Iterate over all beams
    for k in range(measurements.shape[0]):
        r = measurements[k, 0]
        bearing = measurements[k, 1]

        # --------------------------------------------------
        # 1. Determine measurement type
        # --------------------------------------------------

        # Ignore too small values
        if r <= min_sensor_range:
            continue

        # Handle max-range / inf → free space only
        if not np.isfinite(r) or r >= max_sensor_range:
            r_eff = max_sensor_range
            is_hit = False   # NO occupied endpoint
        else:
            r_eff = r
            is_hit = True    # valid obstacle hit

        # --------------------------------------------------
        # 2. Compute endpoint in world coordinates
        # --------------------------------------------------
        phi = heading + bearing

        end_x = x + r_eff * np.cos(phi)
        end_y = y + r_eff * np.sin(phi)

        end_i = int(np.floor((end_y + shift_y) / grid_resolution))
        end_j = int(np.floor((end_x + shift_x) / grid_resolution))

        # If endpoint outside map → skip beam
        if end_i < 0 or end_i >= n_rows or end_j < 0 or end_j >= n_cols:
            beam_out_map_count += 1
            continue

        # --------------------------------------------------
        # 3. Bresenham ray tracing (free + occupied update)
        # --------------------------------------------------
        cell_i = pose_i
        cell_j = pose_j

        dx = abs(end_j - cell_j)
        dy = abs(end_i - cell_i)

        sx = 1 if cell_j < end_j else -1
        sy = 1 if cell_i < end_i else -1

        err = dx - dy

        while True:
            old_val = log_odds_map[cell_i, cell_j]

            # Only update if within bounds
            if min_log_odds < old_val < max_log_odds:

                # Last cell (endpoint)
                if cell_i == end_i and cell_j == end_j:
                    if is_hit:
                        # Only mark occupied if real hit
                        log_odds_map[cell_i, cell_j] = old_val + log_odds_increasing
                    # else: max-range → DO NOTHING (no occupied cell)

                else:
                    # Free space update
                    log_odds_map[cell_i, cell_j] = old_val + log_odds_decreasing

            # Stop at endpoint
            if cell_i == end_i and cell_j == end_j:
                break

            # Bresenham step
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cell_j += sx
            if e2 < dx:
                err += dx
                cell_i += sy

            # Safety check
            if cell_i < 0 or cell_i >= n_rows or cell_j < 0 or cell_j >= n_cols:
                break

    return 0, beam_out_map_count



@njit(cache=True, nogil=True)
def update_map_numba_old_bresenham(
    log_odds_map,
    measurements,
    x,
    y,
    heading,
    shift_x,
    shift_y,
    grid_resolution,
    min_sensor_range,
    max_sensor_range,
    log_odds_decreasing_probability,
    log_odds_increasing_probability,
    min_log_odds,
    max_log_odds,
):
    n_rows, n_cols = log_odds_map.shape

    pose_i = int(np.floor((y + shift_y) / grid_resolution))
    pose_j = int(np.floor((x + shift_x) / grid_resolution))

    if pose_i < 0 or pose_i >= n_rows or pose_j < 0 or pose_j >= n_cols:
        return

    for k in range(measurements.shape[0]):
        r = measurements[k, 0]
        bearing = measurements[k, 1]

        if r <= min_sensor_range or r >= max_sensor_range or not np.isfinite(r):
            continue

        phi = heading + bearing

        reflection_point_x = x + r * np.cos(phi)
        reflection_point_y = y + r * np.sin(phi)

        y_end = int(np.floor((reflection_point_y + shift_y) / grid_resolution))
        x_end = int(np.floor((reflection_point_x + shift_x) / grid_resolution))

        if y_end < 0 or y_end >= n_rows or x_end < 0 or x_end >= n_cols:
            continue

        # ------------------------------------------------------------------
        # Equivalent to your old bresenham_line_drawing()
        # ------------------------------------------------------------------
        y_start = pose_i
        x_start = pose_j

        dx_raw = x_end - x_start
        dy_raw = y_end - y_start

        increment_x = 0
        if dx_raw > 0:
            increment_x = 1
        elif dx_raw < 0:
            increment_x = -1

        increment_y = 0
        if dy_raw > 0:
            increment_y = 1
        elif dy_raw < 0:
            increment_y = -1

        dx = dx_raw
        dy = dy_raw

        if dx < 0:
            dx = -dx
        if dy < 0:
            dy = -dy

        ddx = increment_x
        ddy = increment_y

        if dx > dy:
            pdx = increment_x
            pdy = 0
            slow_direction = dy
            fast_direction = dx
        else:
            pdx = 0
            pdy = increment_y
            slow_direction = dx
            fast_direction = dy

        cell_x = x_start
        cell_y = y_start
        err = fast_direction / 2.0

        # start cell update
        old_log_odds_value = log_odds_map[cell_y, cell_x]

        if old_log_odds_value > min_log_odds and old_log_odds_value < max_log_odds:
            if fast_direction == 0:
                log_odds_map[cell_y, cell_x] = old_log_odds_value + log_odds_increasing_probability
            else:
                log_odds_map[cell_y, cell_x] = old_log_odds_value + log_odds_decreasing_probability

        for step in range(fast_direction):
            err -= slow_direction

            if err < 0:
                err += fast_direction
                cell_x += ddx
                cell_y += ddy
            else:
                cell_x += pdx
                cell_y += pdy

            if cell_y < 0 or cell_y >= n_rows or cell_x < 0 or cell_x >= n_cols:
                break

            old_log_odds_value = log_odds_map[cell_y, cell_x]

            if old_log_odds_value > min_log_odds and old_log_odds_value < max_log_odds:
                if step == fast_direction - 1:
                    log_odds_map[cell_y, cell_x] = old_log_odds_value + log_odds_increasing_probability
                else:
                    log_odds_map[cell_y, cell_x] = old_log_odds_value + log_odds_decreasing_probability



class OGM:
    '''
    Implementation of the occupancy grid mapping algorithm. The map is represented in log Odds space. The map is
    initialized with a prior probability.
    
    Attention! After creating
    the grid map with "create_map()" the map will stay in log Odds space forever. To get the occupancy grid map
    out of the self.log_odds_map, the two methodes "transform_log_odds_map_to_occupancy_grid_map" and 
    "transform_log_odds_map_to_probability_map" need to be used.
    '''
    IDX_X= 0
    IDX_Y= 1
    def __init__(self, map_parameter: List[float], occupancy_parameter: List[float], sensor_parameter: List[float]) -> None:
        '''
        Constructor of the OGM class. Initializes all parameters and variables needed for the algorithm. Also checks 
        if the given parameters are valid and sets them to default values if they are not valid.

        Parameters:
        ----------
        map_parameter: List of float
            All map parameters needed. Containing of the minimum distance to the border of the map, where the map should 
            be extended.
        occupancy_parameter: List of float
            All occupancy parameters needed. Containing of the prior probability, the increasing probability, the decreasing
            probability, the minimum log Odds value, and the maximum log Odds value.
        sensor_parameter: List of float
            All sensor parameters needed. Containing of the minimum and maximum sensor range.

        '''
        # Extract parameter
        self.min_distance_to_border= map_parameter
        prior_probability, increasing_probability, decreasing_probability, self.min_log_odds, self.max_log_odds= occupancy_parameter
        self.min_sensor_range, self.max_sensor_range= sensor_parameter
        self.grid_resolution_m = None

        # Define map
        self.log_odds_map= []                                                          
        self.number_of_cells_x= 0.0
        self.number_of_cells_y= 0.0
        self.left_map_border_m= 0.0
        self.top_map_border_m= 0.0
        self.right_map_border_m= 0.0
        self.bottom_map_border_m= 0.0
        
        # Variables needed for point to grid cell transformation
        self.shift_x= 0
        self.shift_y= 0

        # COunter for counting beam otuside map
        self.beam_out_map_count = 0
        
        # Create OccupancyGrid Message object
        # lom= LogOddsMap()
        self.log_odds_map_msg= LogOddsMap()
        self.log_odds_map_msg.header.frame_id= "log_odds_map"
        
        # Ensure correct prior probability
        if(prior_probability <= 0 or prior_probability > 1.0):                  
            self.log_odds_prior= np.log(prior_probability / (1 - 0.5))    
            rospy.loginfo("\nTHe prior probability must lie between 0 and 1.\n")
            rospy.loginfo("The prior was set to: %f", 0.5)
        else:
            self.log_odds_prior= np.log(prior_probability/(1-prior_probability))    # Calculate log Odds of prior 
        # Ensure correct increasing probability
        if(increasing_probability <= 0 or increasing_probability > 1.0):
            self.log_odds_increasing_probability= np.log(0.65 / 0.35) 
            rospy.loginfo("\nThe increasing probability must lie between 0 and 1.\n")
            rospy.loginfo("The increasing probability was set to: %f", 0.65)
        else:
            self.log_odds_increasing_probability= np.log(increasing_probability / (1 - increasing_probability))
        # Ensure correct decreasing probability
        if(decreasing_probability <= 0 or decreasing_probability >= 1.0):
            self.log_odds_decreasing_probability= np.log(0.35 / 0.65)
            rospy.loginfo("\nThe decreasing probability must lie between 0 and 1.\n")
            rospy.loginfo("The decreasing probability was set to: %f", 0.35)
        else:
            self.log_odds_decreasing_probability= np.log(decreasing_probability / (1 - decreasing_probability))


    def init_map(self, map_width: float, map_height: float, grid_resolution: float) -> None:
        '''Create map and init prior probability'''
        if map_width <= 0 or map_height <= 0 or grid_resolution <= 0:
            rospy.loginfo("\nThe map width, map height, and grid resolution must be positive values.\n")
            rospy.loginfo("The map was not initialized.")
            return

        # Init map parameters
        self.map_width_m = map_width
        self.map_height_m = map_height
        self.grid_resolution_m = grid_resolution

        # Define number of grids in x direction (must be odd value)
        self.number_of_cells_x= ceil(self.map_width_m / self.grid_resolution_m)        
        
        # Check for odds number of grid cells
        if(not (self.number_of_cells_x % 2)):
            self.number_of_cells_x+= 1
        
        # Update map width
        self.map_width_m= self.number_of_cells_x * self.grid_resolution_m
        
        # Define number of grids in y direction(must be odd value)
        self.number_of_cells_y= ceil(self.map_height_m / self.grid_resolution_m)        
        
        # Check for odds number of grid cells
        if(not (self.number_of_cells_y % 2)):
            self.number_of_cells_y+= 1
        
        # Update map Height
        self.map_height_m= self.number_of_cells_y * self.grid_resolution_m
        
        # Create map and initialize prior probability 
        self.log_odds_map= np.full((self.number_of_cells_y, self.number_of_cells_x), self.log_odds_prior)
        
        # Init variables needed transformation (point -> cell)
        self.shift_x= self.map_width_m / 2
        self.shift_y= self.map_height_m / 2
        # Init OccupancyGrid message
        self.update_log_odds_message()
        
        # Define the border values for the map
        half_map_width= self.map_width_m / 2.0
        half_map_height= self.map_height_m / 2.0
        self.left_map_border_m= -half_map_width 
        self.top_map_border_m= half_map_height
        self.right_map_border_m= half_map_width
        self.bottom_map_border_m= - half_map_height        

        rospy.loginfo("An empty map was successfully initialized from the given parameters.")
        rospy.loginfo(
            f"Map width= {self.map_width_m}, Map height= {self.map_height_m},"
            f" Number of cells in x direction= {self.number_of_cells_x},"
            f" Number of cells in y direction= {self.number_of_cells_y}"
        )


    def init_map_from_map(self, log_odds_map: np.ndarray, grid_resolution: float) -> None:
        '''Create the map from a given map.'''
        self.grid_resolution_m = grid_resolution

        if log_odds_map is None:
            rospy.loginfo("The given map is None. The map was not initialized.")
            return

        self.log_odds_map = np.array(log_odds_map, copy=True)

        pad_y = 1 if self.log_odds_map.shape[0] % 2 == 0 else 0
        pad_x = 1 if self.log_odds_map.shape[1] % 2 == 0 else 0

        if pad_y or pad_x:
            self.log_odds_map = np.pad(
                self.log_odds_map,
                pad_width=((pad_y, 0), (pad_x, 0)),
                mode="constant",
                constant_values=self.log_odds_prior,
            )

        self.number_of_cells_y, self.number_of_cells_x = self.log_odds_map.shape

        self.map_width_m = self.number_of_cells_x * self.grid_resolution_m
        self.map_height_m = self.number_of_cells_y * self.grid_resolution_m

        self.shift_x = self.map_width_m / 2
        self.shift_y = self.map_height_m / 2

        self.update_log_odds_message()

        half_map_width = self.map_width_m / 2.0
        half_map_height = self.map_height_m / 2.0
        self.left_map_border_m = -half_map_width
        self.top_map_border_m = half_map_height
        self.right_map_border_m = half_map_width
        self.bottom_map_border_m = -half_map_height

        rospy.loginfo("The map was successfully initialized from the given map.")
        rospy.loginfo(
            f"Map width= {self.map_width_m}, Map height= {self.map_height_m},"
            f" Number of cells in x direction= {self.number_of_cells_x},"
            f" Number of cells in y direction= {self.number_of_cells_y}"
        )


    def init_map_from_map_with_origin(
        self,
        log_odds_map: np.ndarray,
        grid_resolution: float,
        origin_x: float,
        origin_y: float,
    ):
        self.log_odds_map = np.array(log_odds_map, copy=True)
        self.grid_resolution_m = float(grid_resolution)

        self.number_of_cells_y, self.number_of_cells_x = self.log_odds_map.shape

        self.map_width_m = self.number_of_cells_x * self.grid_resolution_m
        self.map_height_m = self.number_of_cells_y * self.grid_resolution_m

        self.shift_x = -float(origin_x)
        self.shift_y = -float(origin_y)

        self.left_map_border_m = origin_x
        self.bottom_map_border_m = origin_y
        self.right_map_border_m = origin_x + self.map_width_m
        self.top_map_border_m = origin_y + self.map_height_m

        self.update_log_odds_message()

        rospy.loginfo("The map was successfully initialized from the given map.")
        rospy.loginfo(
            f"Map width= {self.map_width_m}, Map height= {self.map_height_m},"
            f" Number of cells in x direction= {self.number_of_cells_x},"
            f" Number of cells in y direction= {self.number_of_cells_y}"
            f" Origin x= {origin_x}, Origin y= {origin_y}"
        )


    def get_log_odds_map(self) -> np.ndarray:
        '''Returns the grid map in log odds form.'''
        return self.log_odds_map


    def get_map_meta(self):
        '''
        Returns the log odds map metadata as a dictionary.
        '''
        log_odds_map_meta= {
            "map_width_m": self.map_width_m,
            "map_height_m": self.map_height_m,
            "grid_resolution_m": self.grid_resolution_m,
            "number_of_cells_x": self.number_of_cells_x,
            "number_of_cells_y": self.number_of_cells_y,
            "left_map_border_m": self.left_map_border_m,
            "top_map_border_m": self.top_map_border_m,
            "right_map_border_m": self.right_map_border_m,
            "bottom_map_border_m": self.bottom_map_border_m,
            "shift_x": self.shift_x,
            "shift_y": self.shift_y
        }
        return log_odds_map_meta
    
    
    def get_log_odds_map_object(self) -> LogOddsMap:
        '''
        Returns a log odds map message object containing the map and the map metadata.
        '''
        # Copy the logOdds map to the message
        self.log_odds_map_msg.data= self.log_odds_map.ravel()
        # generate timestamp
        self.log_odds_map_msg.header.stamp= rospy.Time.now()        
        return self.log_odds_map_msg

    
    def extend_map(self, direction: str, distance: float) -> Tuple[int, bool]:
        '''
        Extends the map in the specified direction by the given distance.

        Parameters
        ----------
        direction: str
            The direction in which to extend the map. Can be 'l' (left), 'r' (right), 'b' (bottom), or 't' (top).
        distance: float
            The distance by which to extend the map in meters.
        
        Returns
        -------
        Tuple[int, bool]
            A tuple containing the number of cells the map was extended by and a boolean indicating if the extension was successful.        
        '''
        # Calculate number of cells to extend
        number_of_cells= ceil(distance / self.grid_resolution_m)     
        was_extension_successfull= True   
        old_log_odds_map= np.copy(self.log_odds_map)
        # Extend map on the left
        if(direction == "l"):
            self.number_of_cells_x+= number_of_cells       
            if(not (self.number_of_cells_x % 2)):
                self.number_of_cells_x+= 1   
                number_of_cells+= 1       
            # Update map parameter
            self.map_width_m= self.number_of_cells_x * self.grid_resolution_m 
            extended_distance= number_of_cells * self.grid_resolution_m
            self.shift_x+= extended_distance
            self.left_map_border_m-= extended_distance
            # Create new map 
            self.log_odds_map= np.full((self.number_of_cells_y, self.number_of_cells_x), self.log_odds_prior)
            # Copy old map to the right pose, of the new map
            self.log_odds_map[:, number_of_cells:]= old_log_odds_map    
        # Extend map on the right
        elif(direction == "r"):
            old_number_of_cells_x= self.number_of_cells_x
            self.number_of_cells_x+= number_of_cells
            if(not (self.number_of_cells_x % 2)):
                self.number_of_cells_x+= 1  
                number_of_cells+=1   
            # Update map parameter         
            self.map_width_m= self.number_of_cells_x * self.grid_resolution_m               
            self.right_map_border_m+= number_of_cells * self.grid_resolution_m
            # Create new map 
            self.log_odds_map= np.full((self.number_of_cells_y, self.number_of_cells_x), self.log_odds_prior)
            # Copy old map to the right pose, of the new map
            self.log_odds_map[:, :old_number_of_cells_x]= old_log_odds_map
        # Extend map on the bottom
        elif(direction == "b"):
            self.number_of_cells_y+= number_of_cells
            if(not (self.number_of_cells_y % 2)):
                self.number_of_cells_y+= 1  
                number_of_cells+= 1    
            # Update map parameter 
            self.map_height_m= self.number_of_cells_y * self.grid_resolution_m      
            extended_distance= number_of_cells * self.grid_resolution_m
            self.shift_y+= extended_distance
            self.bottom_map_border_m-= extended_distance
            # Create new map 
            self.log_odds_map= np.full((self.number_of_cells_y, self.number_of_cells_x), self.log_odds_prior)
            # Copy old map to the right pose, of the new map
            self.log_odds_map[number_of_cells:, :]= old_log_odds_map
        # Extend map on the top
        elif(direction == "t"):
            old_number_of_cells_y= self.number_of_cells_y
            self.number_of_cells_y+= number_of_cells
            if(not (self.number_of_cells_y % 2)):
                self.number_of_cells_y+= 1                
                number_of_cells+= 1
            # Update map parameter 
            self.map_height_m= self.number_of_cells_y * self.grid_resolution_m
            self.top_map_border_m+= number_of_cells * self.grid_resolution_m
            # Create new map 
            self.log_odds_map= np.full((self.number_of_cells_y, self.number_of_cells_x), self.log_odds_prior)
            # Copy old map to the right pose, of the new map
            self.log_odds_map[:old_number_of_cells_y, :]= old_log_odds_map
        else:
            was_extension_successfull= False
        return number_of_cells, was_extension_successfull


    def map_extension_if_necessary(self, pose: Tuple[float, float, float]) -> bool:
        '''
        Checks if the map needs to be extended based on the current robot pose.
        '''
        x, y, theta= pose        
        extension_needed= False
        # Check if map needed to be extended on the left side 
        if((x - self.min_distance_to_border) < self.left_map_border_m):
            # rospy.loginfo("position= %f, %f", x, y)
            # rospy.loginfo("")
            self.extend_map(direction='l', distance= self.min_distance_to_border)
            extension_needed= True
            rospy.loginfo("left side extended")
        # Check if map needed to be extended on the right side 
        elif((x + self.min_distance_to_border) > self.right_map_border_m):
            self.extend_map(direction='r', distance= self.min_distance_to_border)
            extension_needed= True
            rospy.loginfo("right side extended")
        # Check if map needed to be extended on the bottom side 
        if((y - self.min_distance_to_border) < self.bottom_map_border_m):
            self.extend_map(direction='b', distance= self.min_distance_to_border)
            extension_needed= True
            rospy.loginfo("bottom extended")
        # Check if map needed to be extended on the top side 
        elif((y + self.min_distance_to_border) > self.top_map_border_m):            
            self.extend_map(direction='t', distance= self.min_distance_to_border)
            extension_needed= True
            rospy.loginfo("top extended")
        if(extension_needed):
            self.update_log_odds_message()
        return extension_needed


    def update_log_odds_message(self) -> None:
        '''
        Updates the log Odds map message with the current map data and metadata. This is needed to convert the array to
        ROS message format at any time! 
        '''
        self.log_odds_map_msg.info.width= self.number_of_cells_x
        self.log_odds_map_msg.info.height= self.number_of_cells_y
        # origin_x, origin_y= self.transform_grid_cell_to_point((0, 0))
        self.log_odds_map_msg.info.origin.position.x= -self.shift_x
        self.log_odds_map_msg.info.origin.position.y= -self.shift_y
        self.log_odds_map_msg.info.origin.orientation.w = 1.0
        self.log_odds_map_msg.info.resolution= self.grid_resolution_m
        self.log_odds_map_msg.header.frame_id = "map"


    #_______________________________________________________________________________________________________________
    # Transformations
    #_______________________________________________________________________________________________________________

    @staticmethod
    def log_odds_to_probability(log_odds: float) -> float:
        '''Calculates the probability according to the given log Odds value.'''
        log_odds_exp= exp(log_odds)
        return log_odds_exp / (1+ log_odds_exp)
    

    @staticmethod
    def transform_occupany_map_to_log_odds_map(ogm: np.ndarray, occ_params: tuple, log_odds_param: tuple) -> float:
        '''
        Transforms the given occupancy grid map to a log Odds map.  
        '''
        # Extract parameters
        occ, free = occ_params
        log_odds_occ, log_odds_free, log_odds_unknown = log_odds_param 

        # Assign log odds values to log odds map
        log_odds_map = np.full_like(ogm, log_odds_unknown, dtype=float)
        log_odds_map[ogm == occ] = log_odds_occ
        log_odds_map[ogm == free] = log_odds_free

        return log_odds_map
    

    @staticmethod
    def probability_to_occupancy(probability: float) -> float:
        occupancy_value= 0.0
        if(probability < 0.5):
            occupancy_value= 0.0
        elif(probability == 0.5):
            occupancy_value= 0.5
        else:
            occupancy_value= 1.0
        return occupancy_value


    @staticmethod
    def transform_log_odds_map_to_probability_map(log_odds_map: np.ndarray) -> np.ndarray:
        '''Transfers the map from the log Odds space to the 
        probability space and returns the transformened map.'''
        probability_map= np.copy(log_odds_map)
        map_shape= np.shape(probability_map)
        for i in range(map_shape[0]):
             for j in range(map_shape[1]):
                #  probability_map[i][j]= self.log_odds_to_probability(self.log_odds_map[i][j])
                probability_map[i][j]= OGM.log_odds_to_probability(log_odds_map[i][j])
        return np.copy(probability_map)

    
    @staticmethod
    def transform_probability_map_to_occupancy_map(probability_grid_map: np.ndarray) -> np.ndarray:
        '''Transforms the given grid map from probability space to occupancy space.'''
        occupancy_grid_map= np.copy(probability_grid_map)
        map_shape= np.shape(probability_grid_map)
        # Round probability of each grid cell to 0, 1, or 0.5
        for i in range(map_shape[0]):
            for j in range(map_shape[1]):
                occupancy_grid_map[i][j]= OGM.probability_to_occupancy(probability_grid_map[i][j])
        return occupancy_grid_map

    
    @staticmethod
    def transform_log_odds_map_to_occupancy_grid_map(log_odds_map: np.ndarray) -> np.ndarray:
        '''Transforms the given map from log odds space to occupancy space.'''
        probability_grid_map= OGM.transform_log_odds_map_to_probability_map(log_odds_map)
        occupancy_grid_map= OGM.transform_probability_map_to_occupancy_map(probability_grid_map)
        return occupancy_grid_map


    def transform_point_to_grid_cell(self, point: Tuple[float, float]) -> Tuple[int, int]:
        '''Transforms an (x, y) point to the array access indices (i, j for row, column). '''
        x, y = point
        x_shifted= x + self.shift_x
        y_shifted= y + self.shift_y
        i= floor(y_shifted/self.grid_resolution_m)
        j= floor(x_shifted/self.grid_resolution_m)
        return (i, j)


    def transform_grid_cell_to_point(self, grid_cell: Tuple[int, int]) -> Tuple[float, float]:
        '''Transforms the given grid cell (i, j) to a (x, y) point in the real world.'''
        i, j= grid_cell
        x= j * self.grid_resolution_m - self.shift_x + self.grid_resolution_m/2
        y= i * self.grid_resolution_m - self.shift_y + self.grid_resolution_m/2
        return (x, y)    


    #_______________________________________________________________________________________________________________
    # Main Algorithm
    #_______________________________________________________________________________________________________________

    def find_reflecting_grid_cell(self, measurement: Tuple[float, float], pose: Tuple[float, float, float]) -> Optional[Tuple[int, int]]:
        '''
        Gets a (range, bearing) measurement and a (x, y, heading) pose and calculates the indices of the reflecting grid
        cell. Also checks if the measurement range is in the area of the sensor range and if the range is infinite. If 
        there is no plausible measurement, then the function return None, otherwise the reflected grid cell.
        '''
        x, y, heading= pose
        range, bearing= measurement
        reflecting_cell= ()
        # Check if range is in max sensor range or infinite
        if(range <= self.min_sensor_range or range >= self.max_sensor_range or not isfinite(range)) :
            # There is no reflecting cell
            reflecting_cell= None
        else: 
            # Ensure angles between -pi and pi
            # bearing= atan2(sin(bearing), cos(bearing))
            # heading= atan2(sin(heading), cos(heading))
            # # Calculate x,y-position of reflected beam.
            phi = heading + bearing
            # phi = atan2(sin(phi), cos(phi))
            reflection_point_x= x + range * cos(phi)
            reflection_point_y= y + range* sin(phi)
            # Transfrom cell coordinates to cell indices.
            reflecting_cell= self.transform_point_to_grid_cell((reflection_point_x, reflection_point_y))
        return reflecting_cell

    
    @staticmethod
    def bresenham_line_drawing(start_grid_idx: Tuple[int, int], end_grid_idx: Tuple[int, int]) -> List[Tuple[int, int]]:
        '''
        Calculates all cell indices between start_grid_idx and end_grid_idx cell. Input values are indices of first and 
        last grid (line, column) (assuming integers).
        '''
        #  y= lines, x = column 
        y_start, x_start= start_grid_idx
        y_end, x_end= end_grid_idx
        affected_cells= []
        # Determine if slope is rising or falling -> y increment up or down
        dx= x_end - x_start
        dy= y_end - y_start
        # Define Increments 
        increment_x= np.sign(dx)
        increment_y= np.sign(dy)
        if(dx<0): dx= -dx
        if(dy<0): dy= -dy
        # Set parameters
        ddx= increment_x
        ddy= increment_y
        if(dx > dy):
            pdx= increment_x
            pdy= 0
            slow_direction= dy
            fast_direction= dx
        else:
            pdx= 0 
            pdy= increment_y
            slow_direction= dx
            fast_direction= dy
        # Initialization
        x= x_start
        y= y_start
        err= fast_direction / 2.0
        affected_cells.append(start_grid_idx)
        # Algorithm
        for i in range(fast_direction):
            err-= slow_direction
            if(err < 0):
                err+= fast_direction
                x+= ddx
                y+= ddy
            else:
                x+= pdx
                y+= pdy
            affected_cells.append((y, x))
        return affected_cells   
    

    def update_affected_cells(self, affected_cells: List[Tuple[int, int]]) -> None:
        '''Get's a list of all effected cells by one beam. Decreases the logOdds values for
        all cells before the last cell. Increases the logOdds value for the last, reflecting, 
        cell.'''
        # Update the occupancy probability for each cell affected by the laser beam
        number_of_affected_cells= len(affected_cells)
        for i in range(number_of_affected_cells):
            cell_i, cell_j= affected_cells[i]
            old_log_odds_value= self.log_odds_map[cell_i][cell_j]
            # Bound the log_odds_values
            if(old_log_odds_value <= self.min_log_odds or old_log_odds_value >= self.max_log_odds):
                # self.log_odds_map[cell_i][cell_j]= old_log_odds_value    
                new_log_odds_value= old_log_odds_value
            else:
                # Decrease the occupancy for all cell that the ray passed
                if(i < (number_of_affected_cells - 1)):
                    new_log_odds_value= old_log_odds_value + self.log_odds_decreasing_probability
                # Increase the occupancy for the cell that reflected the cell
                else: 
                    new_log_odds_value= old_log_odds_value + self.log_odds_increasing_probability
            # Update grid cell
            self.log_odds_map[cell_i][cell_j]= new_log_odds_value


    def update_cells(self, start_grid_idx, end_grid_idx):
        y, x = start_grid_idx
        y_end, x_end = end_grid_idx

        dx = abs(x_end - x)
        dy = abs(y_end - y)

        sx = 1 if x < x_end else -1
        sy = 1 if y < y_end else -1

        err = dx - dy

        while True:
            # ---- THIS IS YOUR update_affected_cells LOGIC ----
            old_log_odds_value = self.log_odds_map[y][x]

            if not (old_log_odds_value <= self.min_log_odds or old_log_odds_value >= self.max_log_odds):

                # last cell = reflecting cell
                if (y == y_end and x == x_end):
                    new_log_odds_value = old_log_odds_value + self.log_odds_increasing_probability
                else:
                    new_log_odds_value = old_log_odds_value + self.log_odds_decreasing_probability

                self.log_odds_map[y][x] = new_log_odds_value

            # stop condition
            if y == y_end and x == x_end:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
            

    def update_map(self, measurements: List[Tuple[float, float]], pose: Tuple[float, float, float]) -> None:
        """
        Update the logOdds map by the given pose and laser measurements.
        Numba-optimized version.
        """
        measurements_np = np.asarray(measurements, dtype=np.float64)

        if measurements_np.size == 0:
            return

        # Extract pose
        x, y, heading = pose

        # status, counter = update_map_numba(
        #     self.log_odds_map,
        #     measurements_np,
        #     x,
        #     y,
        #     heading,
        #     self.shift_x,
        #     self.shift_y,
        #     self.grid_resolution_m,
        #     self.min_sensor_range,
        #     self.max_sensor_range,
        #     self.log_odds_decreasing_probability,
        #     self.log_odds_increasing_probability,
        #     self.min_log_odds,
        #     self.max_log_odds,

        # )

        status, counter = update_map_numba_inf_free_space(
            log_odds_map=self.log_odds_map,
            measurements=measurements_np,
            x=x,
            y=y,
            heading=heading,
            shift_x=self.shift_x,
            shift_y=self.shift_y,
            grid_resolution=self.grid_resolution_m,
            min_sensor_range=self.min_sensor_range,
            max_sensor_range=self.max_sensor_range,
            log_odds_decreasing=self.log_odds_decreasing_probability,
            log_odds_increasing=self.log_odds_increasing_probability,
            min_log_odds=self.min_log_odds,
            max_log_odds=self.max_log_odds,
        )


        # status, beam_out_map_count = update_map_numba_unique_cells(
        #     self.log_odds_map,
        #     measurements_np,
        #     x,
        #     y,
        #     heading,
        #     self.shift_x,
        #     self.shift_y,
        #     self.grid_resolution_m,
        #     self.min_sensor_range,
        #     self.max_sensor_range,
        #     self.log_odds_decreasing_probability,
        #     self.log_odds_increasing_probability,
        #     self.min_log_odds,
        #     self.max_log_odds,
        # )

    
    def update_map_copy(self, measurements: List[Tuple[float, float]], pose: Tuple[float, float, float]) -> None:
        '''Update the logOdds map by the given (x, y, heading) pose and (range, bearing) measurements. 
        Bounds the values of the logOdds map.'''
        x, y, heading= pose
        pose_i, pose_j= self.transform_point_to_grid_cell((x, y))
        for m in measurements:
            # Find grid cell that reflected the ray
            r, bearing = m 
            relfecting_cell= self.find_reflecting_grid_cell((r, bearing), pose)
            # Check if there was a reflecting cell 
            if(relfecting_cell):
                # Find all grid cells between pose and reflecting grid cell
                affected_cells= self.bresenham_line_drawing((pose_i, pose_j), (relfecting_cell))
                self.update_affected_cells(affected_cells)
                # self.update_cells(
                #     start_grid_idx=(pose_i, pose_j),
                #     end_grid_idx=(relfecting_cell),
                # )

    #_______________________________________________________________________________________________________________
    # Map extraction
    #_______________________________________________________________________________________________________________

    def get_neighbors(self, cell) -> List[Tuple[int, int]]:
        '''
        Get's the eight neighbors valures of the given cell. 
        '''
        i, j = cell
        sub_map = self.log_odds_map[i-1:i+2, j-1:j+2].copy()
        sub_map = sub_map.ravel()

        neighbors = np.delete(sub_map, 4)

        return neighbors
        

    def cell_inside_map(self, cell):
        '''
        Check if cell is inside the logOdds map. 
        Useful when computing indices and not sure if index is inside arr or not.
        '''
        cell_inside = False
        i, j = cell

        if 0 <= i < self.log_odds_map.shape[0] and 0 <= j < self.log_odds_map.shape[1]:
            cell_inside = True

        return cell_inside
 

    def cell_belongs_to_surface(self, cell, free_thresh=-2.0, min_free_count=2):
        '''
        Check if the given cell belongs to a surface. A cell belongs to a surface if it has at least "min_free_count"
        free neighbors. If all values around are occupied the cell doesn't belong to a surface, even if the cell itself
        is occupied. This is because the cell would be in the middle of an object and not on the surface of an object. 
        '''
        free_count = 0
        belongs_to_surface = False
        
        neighbors = self.get_neighbors(cell)

        for logOdds in neighbors:
            if logOdds < free_thresh:
                free_count += 1

        if free_count >= min_free_count:
            belongs_to_surface = True
        
        return belongs_to_surface


    def extract_map_for_scan_matching(self, pose, radius, delta_r=1.0, occ_thresh=2.0) -> np.ndarray:
        '''
        This method extracts a part of the map which is used as a target pointcloud for scan matching. 
        '''
        # TODO: Ensure that extracted map size is not too big. Right now it can happen that we are at teh border of the 
        # map array and accidentally jump over and extract a huge part of the map. This is especially the case fpr the method
        # "cell_belongs_to_surface".
        valid_points = []

        # Convert radius into cell numbers
        r_cells = ceil((radius + delta_r) / self.grid_resolution_m)
        r_cells_squared = r_cells * r_cells

        # Transform pose into grid cell
        i_pose, j_pose = self.transform_point_to_grid_cell(pose[:2])

        # With the for loops we define a general square with center cell and it has the size of the radius*2 + 1 
        for di in range(-r_cells, r_cells + 1):
            for dj in range(-r_cells, r_cells + 1):

                # For every cell in the general square we check if the cell is inside the radius (from the center point)
                # Using squared values to avoid sqrt -> faster
                # Skip if not inside circle area
                if di * di + dj * dj > r_cells_squared:
                    continue
                
                # Compute the actual cell indices in the map/array from our general square and the center point
                i = i_pose + di
                j = j_pose + dj

                # Check if cell is indeed inside our map 
                inside_map = self.cell_inside_map((i, j))
                if not inside_map:
                    continue
                
                # Check if cell is occupied -> extract point coordinates
                if self.log_odds_map[i, j] < occ_thresh:
                    continue

                # Check if cell belongs to surface
                belongs_to_surface = self.cell_belongs_to_surface(
                    cell=(i,j),
                    free_thresh=-2.0,
                    min_free_count=2,
                )
                if not belongs_to_surface:
                    continue
                
                # Transform cell to point and append 
                x, y = self.transform_grid_cell_to_point((i, j))
                valid_points.append((x, y))

        return np.copy(valid_points)    


    def extract_map_for_scan_matching_numba(
            self,
            pose,
            radius,
            delta_radius=1.0,
            occ_thresh=2.0,
            surface_radius_m : float = 0.1,
            min_free_ratio: float = 0.25
    ) -> np.ndarray:
        '''
        Extracts the map for scan matching. This variant is speed optimized using numba.

        Parameters
        ----------
        pose: Tuple[float, float, float]
            The pose of the robot (x, y, heading) for which the map should be extracted.
        radius: float
            The radius around the robot pose for which the map should be extracted.
        delta_radius: float, optional
            An additional radius that is added to the given radius to ensure that enough points are extracted for scan matching. Default is 1.0.
        occ_thresh: float, optional
            The log Odds threshold for a cell to be considered occupied. Default is 2.0.
        surface_radius_m: float, optional
            The radius around a map point to consider for surface validation. Default is 0.1.
        min_free_ratio: float, optional
            The minimum ratio of free cells around a occ map point to be valid. We assume those point lies on a surface.
        
        Returns
        -------
        np.ndarray
            An array of shape (N, 2) containing the (x, y) coordinates of the valid points in the map for scan matching.
        '''
        # Transfer map extraction parameters from continuous space to discrete space
        r_cells = int(ceil((radius + delta_radius) / self.grid_resolution_m))
        r_cells_sq = int(r_cells ** 2) 

        # Compute surface radius in grid cells (floating point safe)
        surface_radius_cells = int(ceil(surface_radius_m / self.grid_resolution_m - 1e-12))

        # Compute minimum free cells for surface validation
        surface_window = 2 * surface_radius_cells + 1
        n_cells_surface_window = surface_window**2
        n_neighbors = n_cells_surface_window - 1
        min_free_count = int(ceil(n_neighbors * min_free_ratio))

        i_pose, j_pose = self.transform_point_to_grid_cell(pose[:2])

        # n_cells_great_thres = np.sum(self.log_odds_map > occ_thresh)
        # print("\nNumber of cells above threshold: ", n_cells_great_thres)

        # map_points = extract_map_numba(
        #     self.log_odds_map,
        #     i_pose,
        #     j_pose,
        #     r_cells,
        #     r_cells_sq,
        #     occ_thresh,
        #     -2.0,
        #     2,
        #     self.grid_resolution_m,
        #     self.shift_x,
        #     self.shift_y
        # )

        map_points = extract_map_numba(
            log_odds_map=self.log_odds_map,
            i_pose=i_pose,
            j_pose=j_pose,
            r_cells=r_cells,
            r_cells_sq=r_cells_sq,
            surface_r_cells=surface_radius_cells,
            occ_thresh=occ_thresh,
            free_thresh=-2.0,
            min_free_count=min_free_count,
            grid_res=self.grid_resolution_m,
            shift_x=self.shift_x,
            shift_y=self.shift_y
        )

        # Filter inf and nan values from pre allocated map points -> Only keep actual map points
        map_points = map_points[np.all(np.isfinite(map_points), axis=1)]
        
        return map_points
    

    def extract_submap(self, center_cell, height, width):
        """
        Extract a rectangular submap from log_odds_map for testing! Not access secured, make sure to use 
        valid cell, height and width values!

        Parameters:
            center_cell (tuple): (i, j) center index
            height (int): number of cells in y-direction
            width (int): number of cells in x-direction

        Returns:
            submap (np.ndarray): sliced map
            (i_min, i_max, j_min, j_max): indices in original map
        """

        # Get map and 
        
        n_rows, n_cols = self.log_odds_map.shape

        i_center, j_center = center_cell

        # half sizes
        h_half = height // 2
        w_half = width // 2

        # bounds
        i_min = max(0, i_center - h_half)
        i_max = min(n_rows, i_center + h_half)

        j_min = max(0, j_center - w_half)
        j_max = min(n_cols, j_center + w_half)

        submap = self.log_odds_map[i_min:i_max, j_min:j_max]

        return submap, (i_min, i_max, j_min, j_max)
    

    #_______________________________________________________________________________________________________________
    # Grid Cell manipulation
    #_______________________________________________________________________________________________________________

    def colorize_grid_black(self, grid_cell_indices: Tuple[int, int]) -> None:
        '''For testing. Change the color of the given grid cell to black.'''
        # Define log Odds value that correspond's to black
        logOdds_one= self.max_log_odds                               
        grid_idx_x, grid_idx_y= grid_cell_indices
        self.log_odds_map[grid_idx_x][grid_idx_y]= logOdds_one

    
    def colorize_grid_white(self, grid_cell_indices: Tuple[int, int]) -> None:
        '''For testing. Change the color of the given grid cell to white.'''
        # Define log Odds value that correspond's to white
        logOdds_zero= self.min_log_odds                               
        grid_idx_x, grid_idx_y= grid_cell_indices
        self.log_odds_map[grid_idx_x][grid_idx_y]= logOdds_zero


    def change_grid_cell_value(self, grid_cell_indices: Tuple[int, int], value: float) -> None:
        '''Changes the value of the given grid to the given value.'''
        grid_idx_x, grid_idx_y= grid_cell_indices
        self.log_odds_map[grid_idx_x][grid_idx_y]= value