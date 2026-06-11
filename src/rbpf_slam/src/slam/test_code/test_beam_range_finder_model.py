#!/usr/bin/env python3
import math
import numpy as np

from ..rbpf.beam_range_finder_model import (
    BeamRangeFinderModel,
    _beam_model_prob,
)


class DummyOGM:
    def __init__(
        self,
        log_odds_map,
        shift_x=5.0,
        shift_y=5.0,
        grid_resolution_m=1.0,
        min_sensor_range=0.1,
        max_sensor_range=4.0,
    ):
        self.log_odds_map = np.asarray(log_odds_map, dtype=np.float64)
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.grid_resolution_m = grid_resolution_m
        self.min_sensor_range = min_sensor_range
        self.max_sensor_range = max_sensor_range

    def return_log_odds_map(self):
        return self.log_odds_map


def make_model():
    return BeamRangeFinderModel(
        occ_thresh=1.4,
        free_thresh=-1.4,
        unknown_ratio_thresh=0.30,
        known_free_ratio_thresh=0.70,

        sigma_hit=0.15,
        lambda_short=0.20,
        w_hit=0.70,
        w_short=0.10,
        w_max=0.10,
        w_rand=0.10,

        p_unknown=0.20,
        p_out_of_map=0.10,
        p_unexpected_known_free=0.03,
        p_pred_below_min=0.02,

        alpha_meas=1.0,      # important for easier expected-value tests
        beam_step=1,
        eps=1e-12,
    )


def assert_finite_log(result, name):
    ll = result["log_likelihood"]
    assert np.isfinite(ll), f"{name}: log_likelihood is not finite: {ll}"


def assert_close(a, b, tol=1e-9, name=""):
    assert abs(a - b) <= tol, f"{name}: expected {b}, got {a}"


def test_empty_scan_is_neutral():
    model = make_model()
    ogm = DummyOGM(np.zeros((10, 10)))

    result = model.likelihood(
        pose=(0.0, 0.0, 0.0),
        measurements=[],
        ogm=ogm,
    )

    assert_close(result["log_likelihood"], 0.0, name="empty scan log")
    assert result["valid_beam_count"] == 0


def test_invalid_and_too_close_beams_are_skipped():
    model = make_model()
    ogm = DummyOGM(np.zeros((10, 10)))

    result = model.likelihood(
        pose=(0.0, 0.0, 0.0),
        measurements=[
            (float("nan"), 0.0),
            (0.05, 0.0),       # below min range 0.1
        ],
        ogm=ogm,
    )

    assert_close(result["log_likelihood"], 0.0, name="only skipped beams log")
    assert result["valid_beam_count"] == 0
    assert result["skipped_beam_count"] == 2


def test_pose_outside_map_gets_strong_finite_penalty():
    model = make_model()
    ogm = DummyOGM(np.zeros((10, 10)))

    result = model.likelihood(
        pose=(100.0, 100.0, 0.0),
        measurements=[(2.0, 0.0)],
        ogm=ogm,
    )

    assert np.isfinite(result["log_likelihood"])
    assert result["log_likelihood"] < -1e9


def test_map_hit_correct_range_better_than_wrong_range():
    model = make_model()

    # Pose (0,0) -> cell (5,5)
    # Beam bearing 0 goes along +x.
    # Occupied cell at i=5, j=7 -> world center x=2.5, y=0.5
    # Expected range is roughly sqrt(2.5^2 + 0.5^2), because cell center is used.
    grid = np.zeros((10, 10))
    grid[5, 7] = 2.0
    ogm = DummyOGM(grid)

    # Compute with two measurements: one near expected, one clearly wrong.
    good = model.likelihood(
        pose=(0.0, 0.0, 0.0),
        measurements=[(2.55, 0.0)],
        ogm=ogm,
    )

    bad = model.likelihood(
        pose=(0.0, 0.0, 0.0),
        measurements=[(3.8, 0.0)],
        ogm=ogm,
    )

    assert_finite_log(good, "good map hit")
    assert_finite_log(bad, "bad map hit")

    assert good["map_hit_count"] == 1
    assert bad["map_hit_count"] == 1
    assert good["log_likelihood"] > bad["log_likelihood"], (
        f"Expected correct range to be better. good={good['log_likelihood']}, bad={bad['log_likelihood']}"
    )


def test_map_hit_and_measured_max_is_finite():
    model = make_model()

    grid = np.zeros((10, 10))
    grid[5, 7] = 2.0
    ogm = DummyOGM(grid)

    result = model.likelihood(
        pose=(0.0, 0.0, 0.0),
        measurements=[(4.0, 0.0)],   # max range
        ogm=ogm,
    )

    assert_finite_log(result, "map hit measured max")
    assert result["map_hit_count"] == 1


def test_known_free_ray_with_max_range_is_high_probability():
    model = make_model()

    # Known free cells along ray, no occupied cell.
    grid = np.zeros((10, 10))
    grid[5, 6:10] = -2.0
    ogm = DummyOGM(grid)

    max_result = model.likelihood(
        pose=(0.0, 0.0, 0.0),
        measurements=[(4.0, 0.0)],   # sensor also says no obstacle
        ogm=ogm,
    )

    finite_result = model.likelihood(
        pose=(0.0, 0.0, 0.0),
        measurements=[(2.0, 0.0)],   # unexpected obstacle in known-free ray
        ogm=ogm,
    )

    assert_finite_log(max_result, "known free max")
    assert_finite_log(finite_result, "known free finite")

    assert max_result["known_free_ray_count"] == 1
    assert finite_result["known_free_ray_count"] == 1
    assert finite_result["unexpected_known_free_count"] == 1

    assert max_result["log_likelihood"] > finite_result["log_likelihood"], (
        "Known-free + max range should be better than known-free + finite unexpected hit."
    )


def test_unknown_ray_uses_p_unknown():
    model = make_model()

    # All cells unknown = 0.0, no occupied hit.
    grid = np.zeros((10, 10))
    ogm = DummyOGM(grid)

    result = model.likelihood(
        pose=(0.0, 0.0, 0.0),
        measurements=[(2.0, 0.0)],
        ogm=ogm,
    )

    assert_finite_log(result, "unknown ray")
    assert result["unknown_ray_count"] == 1

    expected = math.log(model.p_unknown)
    assert_close(result["log_likelihood"], expected, tol=1e-9, name="unknown ray log")


def test_out_of_map_ray_uses_p_out_of_map():
    model = make_model()

    # Small map. Pose inside, but ray to max range leaves the map.
    grid = np.zeros((3, 3))
    ogm = DummyOGM(
        grid,
        shift_x=1.0,
        shift_y=1.0,
        grid_resolution_m=1.0,
        min_sensor_range=0.1,
        max_sensor_range=10.0,
    )

    result = model.likelihood(
        pose=(0.0, 0.0, 0.0),
        measurements=[(2.0, 0.0)],
        ogm=ogm,
    )

    assert_finite_log(result, "out of map ray")
    assert result["out_of_map_count"] == 1

    expected = math.log(model.p_out_of_map)
    assert_close(result["log_likelihood"], expected, tol=1e-9, name="out of map log")


def test_random_component_is_zero_at_max_range():
    """
    Book detail:
    p_rand = 1 / max_range only for 0 <= z < max_range.
    At z == max_range, p_rand should be 0.
    """

    max_range = 4.0
    eps = 1e-12

    p_finite = _beam_model_prob(
        z=2.0,
        z_exp=3.0,
        measured_max=False,
        max_sensor_range=max_range,
        sigma_hit=0.15,
        lambda_short=0.2,
        w_hit=0.0,
        w_short=0.0,
        w_max=0.0,
        w_rand=1.0,
        eps=eps,
    )

    p_max = _beam_model_prob(
        z=max_range,
        z_exp=max_range,
        measured_max=True,
        max_sensor_range=max_range,
        sigma_hit=0.15,
        lambda_short=0.2,
        w_hit=0.0,
        w_short=0.0,
        w_max=0.0,
        w_rand=1.0,
        eps=eps,
    )

    assert_close(p_finite, 1.0 / max_range, name="p_rand finite")
    assert_close(p_max, eps, name="p_rand at max should be only eps floor")


def test_no_result_contains_inf_or_nan_for_common_cases():
    model = make_model()

    grids = []

    # Unknown map
    grids.append(np.zeros((10, 10)))

    # Known free map
    g_free = np.zeros((10, 10))
    g_free[5, 6:10] = -2.0
    grids.append(g_free)

    # Occupied map
    g_occ = np.zeros((10, 10))
    g_occ[5, 7] = 2.0
    grids.append(g_occ)

    measurements = [
        [(2.0, 0.0)],
        [(4.0, 0.0)],
        [(float("inf"), 0.0)],
        [(float("nan"), 0.0), (0.05, 0.0), (2.0, 0.0)],
    ]

    for gi, grid in enumerate(grids):
        ogm = DummyOGM(grid)

        for mi, meas in enumerate(measurements):
            result = model.likelihood(
                pose=(0.0, 0.0, 0.0),
                measurements=meas,
                ogm=ogm,
            )

            assert np.isfinite(result["log_likelihood"]), (
                f"grid={gi}, meas={mi}, result={result}"
            )


def run_all_tests():
    test_empty_scan_is_neutral()
    test_invalid_and_too_close_beams_are_skipped()
    test_pose_outside_map_gets_strong_finite_penalty()
    test_map_hit_correct_range_better_than_wrong_range()
    test_map_hit_and_measured_max_is_finite()
    test_known_free_ray_with_max_range_is_high_probability()
    test_unknown_ray_uses_p_unknown()
    test_out_of_map_ray_uses_p_out_of_map()
    test_random_component_is_zero_at_max_range()
    test_no_result_contains_inf_or_nan_for_common_cases()

    print("All BeamRangeFinderModel tests passed.")


if __name__ == "__main__":
    run_all_tests()