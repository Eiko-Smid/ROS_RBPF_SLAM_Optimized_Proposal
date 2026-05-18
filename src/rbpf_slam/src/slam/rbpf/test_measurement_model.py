#!/usr/bin/env python3

# import debugpy
# debugpy.listen(("localhost", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()


from typing import Tuple, List

import numpy as np

from .likelihood_filed_model import LikelihoodFiledModel
from .proposal import ProposalEstimator
from .scan_match_factory import (
    ICPParams,
    MapParameter,
    OccupancyParams,
    RobotParams,
    ScanMatchFactory,
    ScanMatcherParams,
    SensorParams,
)

from ..infrastructure.defs import Pose2D


def _compute_wheel_separation() -> float:
    h_chassis = 0.15
    dist_chassis_to_ground = h_chassis / 5
    r_wheel = h_chassis / 2 + dist_chassis_to_ground
    w_wheel = 0.3 * r_wheel
    r_chassis = 0.25
    return 2 * r_chassis + w_wheel


def init_scan_matcher():
    """Build a fully initialized scan matcher with sane defaults for measurement-model tests."""
    return ScanMatchFactory.build(
        occ_param=OccupancyParams(
            min_distance_to_border=8.0,
            prior_probability=0.5,
            increasing_probability=0.7,
            decreasing_probability=0.30,
            min_log_odds=-5.0,
            max_log_odds=5.0,
        ),
        sens_params=SensorParams(
            min_sensor_range=0.1,
            max_sensor_range=10.0,
        ),
        map_param=MapParameter(
            map_width=15.0,
            map_height=15.0,
            grid_resolution_m=0.1,
        ),
        icp_params=ICPParams(
            max_n_points=400,
            max_correspondence_distance=0.6,
            neighbors_pca=10,
            max_iterations=5,
            epsilon_rel=1e-3,
            no_improvement_limit=3,
            min_error=5e-4,
            min_dtrans=1e-3,
            min_drot=1e-2,
            min_points=20,
            min_corresp=15,
            min_hessian_rank=3,
            max_hessian_condition=1e8,
            max_translation_jump=0.8,
            max_rotation_jump=np.deg2rad(120.0),
            max_acceptable_mean_error=0.15,
        ),
        robo_param=RobotParams(
            wheel_separation=_compute_wheel_separation(),
        ),
        sm_params=ScanMatcherParams(
            occ_thres=1.2,
            delta_r=0.6,
        ),
    )


def init_measurement_model(sigma: float = 0.2) -> LikelihoodFiledModel:
    return LikelihoodFiledModel(sigma=sigma)


def init_proposal() -> ProposalEstimator:
    return ProposalEstimator()


def init_test_components(
    sigma_measurement: float = 0.2,
):
    """Create only the components required for measurement-likelihood/proposal tests."""
    scan_matcher = init_scan_matcher()
    measurement_model = init_measurement_model(sigma=sigma_measurement)
    proposal = init_proposal()


    return scan_matcher, measurement_model, proposal




def def_measurements():
    measurements = [
        (2.0, np.radians(-90.0)),
        (2.0, np.radians(0.0)),
        (2.0, np.radians(90.0)),
    ]

    return measurements


def def_map_points(min_xj: Pose2D, measurements: List[Tuple[float, float]]):
    # Compute map points from measurement for true xj
    x, y, theta = min_xj
    map_points = []

    for r, b in measurements:
        # Compute map points
        mpx = x + r * np.cos(theta + b)
        mpy = y + r * np.sin(theta + b)

        map_points.append((mpx, mpy))

    return map_points


def points_equal(points_a, points_b, atol=1e-8, rtol=1e-6):
    points_a = np.asarray(points_a)
    points_b = np.asarray(points_b)

    if points_a.shape != points_b.shape:
        return False

    points_a = points_a[np.lexsort((points_a[:, 1], points_a[:, 0]))]
    points_b = points_b[np.lexsort((points_b[:, 1], points_b[:, 0]))]

    return np.allclose(points_a, points_b, atol=atol, rtol=rtol)


def main():
    # Init components
    scan_matcher, measurement_model, proposal = init_test_components()
    print("Initialized components for measurement model test.")

    # TODO: Also do this with xj slightly off later on 
    # Define true pose -> true_xj
    
    # init params
    sigma_xy: float = 0.05
    sigma_theta: float = 0.02
    n_samples_dir: int = 3

    # Def samples space
    scan_match_pose: Pose2D = (0.0, 0.0, 0.0)
    samples, n_xj = proposal.sample_poses_deterministic(
        pose=scan_match_pose,
        sigma_xy=sigma_xy,
        sigma_theta=sigma_theta,
        n_samples_dir=n_samples_dir,
    )

    # Define true pose = xj top left
    smx, smy, smtheta = scan_match_pose
    true_pose: Pose2D = (
        smx + sigma_xy,
        smy + sigma_xy,
        smtheta,
    )
    print(f"\nTrue pose: {true_pose[0]:.3f}, {true_pose[1]:.3f}, {np.degrees(true_pose[2]):.2f} deg")

    # Check if true pose is part of samples 
    if true_pose not in samples:
        print("True pose is not part of the samples. Consider increasing sigma or n_samples_dir.")
    else:
        print("True pose is part of the samples.")

    # Define measurements and map points
    measurements = def_measurements()
    map_points = def_map_points(true_pose, measurements)

    # Check map size
    map_width = scan_matcher.ogm.map_width_m
    map_height = scan_matcher.ogm.map_height_m
    print(f"Map width: {map_width:.2f} m, height: {map_height:.2f} m")
    if map_width > 15.2 or map_height > 15.2:
        raise ValueError("Map size is larger than expected. Check map parameters.")
    else:
        print("Map size is within expected bounds.")
    
    # Set map points occupied
    for mp in map_points:
        grid_idx_x, grid_idx_y = scan_matcher.ogm.transform_point_to_grid_cell(mp)
        scan_matcher.ogm.colorize_grid_black((grid_idx_x, grid_idx_y))

    # Check if correct map points have been set to occupied
    # Get ogm map
    log_odds_map = scan_matcher.ogm.return_log_odds_map()
    # Find all cells with log odds value is occupied
    occ = scan_matcher.ogm.max_log_odds
    cell_indices = np.argwhere(log_odds_map == occ)
    # Transf indices back to points and check 
    extracted_map_points = []
    for cell_idx in cell_indices:
        mp = scan_matcher.ogm.transform_grid_cell_to_point(cell_idx)
        extracted_map_points.append(mp)

    # Check if values are withing threshold the original ones
    if points_equal(map_points, extracted_map_points):
        print(f"\nAll map points are equal.")
    else: 
        raise ValueError(f"\nExtracted map points do not match original map points. Extracted: {extracted_map_points}, Original: {map_points}")

    # Compute measurement likelihoods for all samples
    


if __name__ == "__main__":
    main()
