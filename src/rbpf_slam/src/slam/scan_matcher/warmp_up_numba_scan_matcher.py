#!/usr/bin/env python3

"""Small correctness-checking warm-ups for the active scan-matcher Numba kernels."""

import numpy as np

from .icp_scan_matching import (
    compute_normals_numba,
    prepare_system_point_to_plane_numba,
)
from .ogm_scan_matching import (
    extract_map_numba,
    update_map_numba_inf_free_space,
)


def warm_up_compute_normals_numba() -> None:
    """Compile ``compute_normals_numba`` and validate horizontal-line normals."""
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float64,
    )
    indices = np.tile(
        np.arange(points.shape[0], dtype=np.int64),
        (points.shape[0], 1),
    )

    normals = compute_normals_numba(points, indices)
    expected_absolute_normals = np.tile(
        np.array([0.0, 1.0], dtype=np.float64),
        (points.shape[0], 1),
    )

    if normals.shape != points.shape:
        raise RuntimeError(
            "compute_normals_numba warm-up failed: "
            f"expected shape {points.shape}, received {normals.shape}."
        )

    if not np.all(np.isfinite(normals)):
        raise RuntimeError(
            "compute_normals_numba warm-up failed: normals contain non-finite values."
        )

    if not np.allclose(
        np.abs(normals),
        expected_absolute_normals,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError(
            "compute_normals_numba warm-up failed: "
            "a horizontal line did not produce vertical unit normals."
        )


def warm_up_prepare_system_point_to_plane_numba() -> None:
    """Compile the point-to-plane system kernel and validate its small system."""
    transformation = np.zeros((3, 1), dtype=np.float64)
    latest_new_data = np.array([[1.0, 1.0]], dtype=np.float64)
    true_data_pointpairs = np.array([[1.0, 0.0]], dtype=np.float64)
    correspondences = np.array([[0, 0]], dtype=np.int64)
    true_data_normals = np.array([[0.0, 1.0]], dtype=np.float64)

    hessian, gradient, squared_error = prepare_system_point_to_plane_numba(
        transformation_parameter=transformation,
        latest_new_data=latest_new_data,
        true_data_pointpairs=true_data_pointpairs,
        correspondences=correspondences,
        true_data_normals=true_data_normals,
    )

    expected_hessian = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    expected_gradient = np.array([[0.0], [0.5], [0.5]], dtype=np.float64)

    if not np.allclose(hessian, expected_hessian, rtol=1e-12, atol=1e-12):
        raise RuntimeError(
            "prepare_system_point_to_plane_numba warm-up failed: "
            "the Hessian does not match the expected point-to-plane system."
        )

    if not np.allclose(gradient, expected_gradient, rtol=1e-12, atol=1e-12):
        raise RuntimeError(
            "prepare_system_point_to_plane_numba warm-up failed: "
            "the gradient does not match the expected point-to-plane system."
        )

    if not np.isclose(squared_error, 1.0, rtol=1e-12, atol=1e-12):
        raise RuntimeError(
            "prepare_system_point_to_plane_numba warm-up failed: "
            f"expected squared error 1.0, received {squared_error}."
        )


def warm_up_update_map_numba_inf_free_space() -> None:
    """Compile the map updater and validate finite-hit and infinite-range beams."""
    finite_hit_map = np.zeros((7, 7), dtype=np.float64)
    finite_measurement = np.array([[2.0, 0.0]], dtype=np.float64)

    status, beam_out_map_count = update_map_numba_inf_free_space(
        log_odds_map=finite_hit_map,
        measurements=finite_measurement,
        x=0.0,
        y=0.0,
        heading=0.0,
        shift_x=3.0,
        shift_y=3.0,
        grid_resolution=1.0,
        min_sensor_range=0.1,
        max_sensor_range=3.0,
        log_odds_decreasing=-0.4,
        log_odds_increasing=0.8,
        min_log_odds=-5.0,
        max_log_odds=5.0,
    )

    expected_finite_hit_map = np.zeros((7, 7), dtype=np.float64)
    expected_finite_hit_map[3, 3] = -0.4
    expected_finite_hit_map[3, 4] = -0.4
    expected_finite_hit_map[3, 5] = 0.8

    if (status, beam_out_map_count) != (0, 0):
        raise RuntimeError(
            "update_map_numba_inf_free_space warm-up failed: "
            f"expected status/counter (0, 0), received {(status, beam_out_map_count)}."
        )

    if not np.allclose(finite_hit_map, expected_finite_hit_map):
        raise RuntimeError(
            "update_map_numba_inf_free_space warm-up failed: "
            "a finite beam did not clear its ray and mark its endpoint occupied."
        )

    infinite_range_map = np.zeros((7, 7), dtype=np.float64)
    infinite_measurement = np.array([[np.inf, 0.0]], dtype=np.float64)

    status, beam_out_map_count = update_map_numba_inf_free_space(
        log_odds_map=infinite_range_map,
        measurements=infinite_measurement,
        x=0.0,
        y=0.0,
        heading=0.0,
        shift_x=3.0,
        shift_y=3.0,
        grid_resolution=1.0,
        min_sensor_range=0.1,
        max_sensor_range=2.0,
        log_odds_decreasing=-0.4,
        log_odds_increasing=0.8,
        min_log_odds=-5.0,
        max_log_odds=5.0,
    )

    expected_infinite_range_map = np.zeros((7, 7), dtype=np.float64)
    expected_infinite_range_map[3, 3] = -0.4
    expected_infinite_range_map[3, 4] = -0.4

    if (status, beam_out_map_count) != (0, 0):
        raise RuntimeError(
            "update_map_numba_inf_free_space warm-up failed for an infinite beam: "
            f"expected status/counter (0, 0), received {(status, beam_out_map_count)}."
        )

    if not np.allclose(infinite_range_map, expected_infinite_range_map):
        raise RuntimeError(
            "update_map_numba_inf_free_space warm-up failed: "
            "an infinite beam did not clear free space without creating an occupied endpoint."
        )


def warm_up_extract_map_numba() -> None:
    """Compile the map extractor and validate one occupied surface point."""
    log_odds_map = np.zeros((5, 5), dtype=np.float64)
    log_odds_map[1:4, 1:4] = -1.0
    log_odds_map[2, 2] = 2.0

    map_points = extract_map_numba(
        log_odds_map=log_odds_map,
        i_pose=2,
        j_pose=2,
        r_cells=1,
        r_cells_sq=1,
        surface_r_cells=1,
        occ_thresh=1.0,
        free_thresh=-0.5,
        min_free_count=1,
        grid_res=0.5,
        shift_x=1.0,
        shift_y=1.0,
    )
    expected_map_points = np.array([[0.25, 0.25]], dtype=np.float64)

    if map_points.shape != expected_map_points.shape:
        raise RuntimeError(
            "extract_map_numba warm-up failed: "
            f"expected shape {expected_map_points.shape}, received {map_points.shape}."
        )

    if not np.allclose(map_points, expected_map_points, rtol=1e-12, atol=1e-12):
        raise RuntimeError(
            "extract_map_numba warm-up failed: "
            f"expected point {expected_map_points.tolist()}, received {map_points.tolist()}."
        )


def warm_up_numba_scan_matcher() -> None:
    """Warm up and validate every Numba kernel used by the scan matcher."""
    print("\nWarming up Numba functions for scan matcher.")
    warm_up_compute_normals_numba()
    warm_up_prepare_system_point_to_plane_numba()
    warm_up_update_map_numba_inf_free_space()
    warm_up_extract_map_numba()
    print("All Numba functions for scan matcher warmed up successfully.")
