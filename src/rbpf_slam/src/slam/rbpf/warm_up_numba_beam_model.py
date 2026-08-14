#!/usr/bin/env python3

"""Small correctness-checking warm-ups for the active beam-model Numba kernels."""

from math import log, pi, sqrt

import numpy as np

from .beam_range_finder_model import (
    _beam_model_prob,
    _raytrace_first_occupied_cell,
    meas_model_likelihood_numba,
    raytracing_log_likelihood_numba,
)


def _create_beam_test_map() -> np.ndarray:
    """Create a small map with one occupied cell two metres from the robot."""
    log_odds_map = np.zeros((7, 7), dtype=np.float64)
    log_odds_map[3, 5] = 2.0
    return log_odds_map


def warm_up_beam_model_prob() -> None:
    """Compile ``_beam_model_prob`` and validate a centered Gaussian hit."""
    probability = _beam_model_prob(
        z=2.0,
        z_exp=2.0,
        measured_max=False,
        max_sensor_range=3.0,
        sigma_hit=1.0,
        lambda_short=1.0,
        w_hit=1.0,
        w_short=0.0,
        w_max=0.0,
        w_rand=0.0,
        eps=1e-12,
    )
    expected_probability = 1.0 / sqrt(2.0 * pi)

    if not np.isclose(
        probability,
        expected_probability,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError(
            "_beam_model_prob warm-up failed: "
            f"expected {expected_probability}, received {probability}."
        )


def warm_up_raytrace_first_occupied_cell() -> None:
    """Compile the raytracer and validate its first occupied-cell result."""
    result = _raytrace_first_occupied_cell(
        log_odds_map=_create_beam_test_map(),
        pose_i=3,
        pose_j=3,
        end_i=3,
        end_j=6,
        occ_thresh=1.0,
        free_thresh=-1.0,
    )
    expected_result = (True, 3, 5, False, 0, 1, 2)

    if result != expected_result:
        raise RuntimeError(
            "_raytrace_first_occupied_cell warm-up failed: "
            f"expected {expected_result}, received {result}."
        )


def warm_up_raytracing_log_likelihood_numba() -> None:
    """Compile the single-pose likelihood kernel and validate a perfect hit."""
    measurements = np.array([[2.0, 0.0]], dtype=np.float64)
    result = raytracing_log_likelihood_numba(
        log_odds_map=_create_beam_test_map(),
        shift_x=3.5,
        shift_y=3.5,
        occ_thresh=1.0,
        free_thresh=-1.0,
        grid_resolution=1.0,
        min_sensor_range=0.1,
        max_sensor_range=3.0,
        measurements=measurements,
        x=0.0,
        y=0.0,
        heading=0.0,
        unknown_ratio_thresh=0.5,
        known_free_ratio_thresh=0.5,
        sigma_hit=1.0,
        lambda_short=1.0,
        w_hit=1.0,
        w_short=0.0,
        w_max=0.0,
        w_rand=0.0,
        p_unknown=0.1,
        p_out_of_map=0.1,
        p_unexpected_known_free=0.1,
        p_pred_below_min=0.1,
        alpha_meas=1.0,
        beam_step=1,
        eps=1e-12,
    )

    expected_log_likelihood = log(1.0 / sqrt(2.0 * pi))
    expected_counters = (1, 1, 0, 0, 0, 0, 0, 0)

    if not np.isclose(result[0], expected_log_likelihood, rtol=1e-12, atol=1e-12):
        raise RuntimeError(
            "raytracing_log_likelihood_numba warm-up failed: "
            f"expected log-likelihood {expected_log_likelihood}, received {result[0]}."
        )

    if not np.isclose(result[1], 0.0, rtol=1e-12, atol=1e-12):
        raise RuntimeError(
            "raytracing_log_likelihood_numba warm-up failed: "
            f"expected zero mean absolute error, received {result[1]}."
        )

    if result[2:] != expected_counters:
        raise RuntimeError(
            "raytracing_log_likelihood_numba warm-up failed: "
            f"expected counters {expected_counters}, received {result[2:]}."
        )


def warm_up_meas_model_likelihood_numba() -> None:
    """Compile the batch likelihood kernel and validate two identical poses."""
    measurements = np.array([[2.0, 0.0]], dtype=np.float64)
    poses = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    result = meas_model_likelihood_numba(
        log_odds_map=_create_beam_test_map(),
        shift_x=3.5,
        shift_y=3.5,
        occ_thresh=1.0,
        free_thresh=-1.0,
        grid_resolution=1.0,
        min_sensor_range=0.1,
        max_sensor_range=3.0,
        measurements=measurements,
        poses=poses,
        unknown_ratio_thresh=0.5,
        known_free_ratio_thresh=0.5,
        sigma_hit=1.0,
        lambda_short=1.0,
        w_hit=1.0,
        w_short=0.0,
        w_max=0.0,
        w_rand=0.0,
        p_unknown=0.1,
        p_out_of_map=0.1,
        p_unexpected_known_free=0.1,
        p_pred_below_min=0.1,
        alpha_meas=1.0,
        beam_step=1,
        eps=1e-12,
    )

    expected_log_likelihood = log(1.0 / sqrt(2.0 * pi))
    expected_log_likelihoods = np.full(2, expected_log_likelihood, dtype=np.float64)
    expected_mean_abs_errors = np.zeros(2, dtype=np.float64)
    expected_counters = (2, 2, 2, 0, 0, 0, 0, 0, 0)

    if not np.allclose(
        result[0],
        expected_log_likelihoods,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError(
            "meas_model_likelihood_numba warm-up failed: "
            f"expected log-likelihoods {expected_log_likelihoods}, received {result[0]}."
        )

    if not np.allclose(
        result[1],
        expected_mean_abs_errors,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError(
            "meas_model_likelihood_numba warm-up failed: "
            f"expected mean errors {expected_mean_abs_errors}, received {result[1]}."
        )

    if result[2:] != expected_counters:
        raise RuntimeError(
            "meas_model_likelihood_numba warm-up failed: "
            f"expected counters {expected_counters}, received {result[2:]}."
        )


def warm_up_numba_beam_model() -> None:
    """Warm up and validate every Numba kernel used by the beam model."""
    print("\nWarming up Numba functions for Measurement model.")
    warm_up_beam_model_prob()
    warm_up_raytrace_first_occupied_cell()
    warm_up_raytracing_log_likelihood_numba()
    warm_up_meas_model_likelihood_numba()
    print("All Numba functions for Measurement model warmed up successfully.\n")
