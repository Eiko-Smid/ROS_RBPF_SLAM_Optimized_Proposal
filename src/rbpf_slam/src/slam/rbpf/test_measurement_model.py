#!/usr/bin/env python3

# import debugpy
# debugpy.listen(("localhost", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()


from typing import Tuple, List
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from ..scan_matcher.scan_matcher import ScanMatcher

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

from ..rbpf.rbpf import MeasurementModelParams

from ..infrastructure.defs import Pose2D
from ..optimize_rbpf.evaluator import RBPFEvaluator


#____________________________________________________________________________________________________
# Dataclasses for test parameters and data
#____________________________________________________________________________________________________

@dataclass
class GmappingMeasurementModelParams:
    usable_range: float = 10.0
    kernel_size: int = 1
    fullness_threshold: float = 1.2
    free_threshold: float = 1.2
    gaussian_sigma: float = 0.05
    free_cell_ratio: float = np.sqrt(2)


@dataclass
class ProposalParams:
    sigma_xy: float = 0.05
    sigma_theta: float = np.radians(2.0)
    n_samples_dir: int = 3


@dataclass
class MeasurementExpParams:
    sensor_params: SensorParams

    robot_params: RobotParams

    scan_matcher_params: ScanMatcherParams
    occupancy_params: OccupancyParams
    map_params: MapParameter
    icp_params: ICPParams

    meas_model_params: MeasurementModelParams
    gmapping_meas_model_params: GmappingMeasurementModelParams

    proposal_params: ProposalParams



#____________________________________________________________________________________________________
# Init params
#____________________________________________________________________________________________________


def _compute_wheel_separation() -> float:
    h_chassis = 0.15
    dist_chassis_to_ground = h_chassis / 5
    r_wheel = h_chassis / 2 + dist_chassis_to_ground
    w_wheel = 0.3 * r_wheel
    r_chassis = 0.25
    return 2 * r_chassis + w_wheel


def init_exp_params() -> MeasurementExpParams:
    return MeasurementExpParams(
        sensor_params = SensorParams(
            min_sensor_range=0.1,
            max_sensor_range=10.0,
        ),

        robot_params = RobotParams(
            wheel_separation=_compute_wheel_separation()
        ),

        scan_matcher_params=ScanMatcherParams(
            occ_thres=1.2,
            delta_r=0.6,
        ),
        occupancy_params=OccupancyParams(
            min_distance_to_border=8.0,
            prior_probability=0.5,
            increasing_probability=0.7,
            decreasing_probability=0.30,
            min_log_odds=-5.0,
            max_log_odds=5.0,
        ),
        map_params=MapParameter(
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
        meas_model_params=MeasurementModelParams(
            sigma_measurement=0.1
        ),
        gmapping_meas_model_params=GmappingMeasurementModelParams(
                usable_range=10.0,
                kernel_size=1,
                fullness_threshold=1.2,
                free_threshold=1.2,
                gaussian_sigma=0.05,
                free_cell_ratio=np.sqrt(2)
        ),
        proposal_params=ProposalParams(
            sigma_xy=0.05,
            sigma_theta=np.radians(2.0),
            n_samples_dir=3,
        ),
    )


def init_scan_matcher(exp_params: MeasurementExpParams = None) -> ScanMatcher:
    """Build a fully initialized scan matcher with sane defaults for measurement-model tests."""
    return ScanMatchFactory.build(
        occ_param=exp_params.occupancy_params,
        sens_params=exp_params.sensor_params,
        map_param=exp_params.map_params,

        icp_params=exp_params.icp_params,
        robo_param=exp_params.robot_params,
        sm_params=exp_params.scan_matcher_params,
    )


def init_measurement_model(exp_params: MeasurementExpParams) -> LikelihoodFiledModel:
    return LikelihoodFiledModel(sigma=exp_params.meas_model_params.sigma_measurement)


def init_proposal() -> ProposalEstimator:
    return ProposalEstimator()


def init_test_components(
    exp_params: MeasurementExpParams,
) -> Tuple[ScanMatchFactory, LikelihoodFiledModel, ProposalEstimator]:
    """Create only the components required for measurement-likelihood/proposal tests."""
    scan_matcher = init_scan_matcher(exp_params=exp_params)
    measurement_model = init_measurement_model(exp_params=exp_params)
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


#____________________________________________________________________________________________________
# Main
#____________________________________________________________________________________________________

def main():

    exp_params = init_exp_params()

    #___________________________________________________________________________________________________
    # Init components
    #___________________________________________________________________________________________________

    scan_matcher, measurement_model, proposal = init_test_components(exp_params=exp_params)
    print("Initialized components for measurement model test.")


    # Define sample space
    scan_match_pose: Pose2D = (0.0, 0.0, 0.0)
    samples, n_xj = proposal.sample_poses_deterministic(
        pose=scan_match_pose,
        sigma_xy=exp_params.proposal_params.sigma_xy,
        sigma_theta=exp_params.proposal_params.sigma_theta,
        n_samples_dir=exp_params.proposal_params.n_samples_dir,
    )

    # Define true pose = xj top left
    smx, smy, smtheta = scan_match_pose
    true_pose: Pose2D = (
        smx + exp_params.proposal_params.sigma_xy,
        smy + exp_params.proposal_params.sigma_xy,
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
        grid_idx_i, grid_idx_j = scan_matcher.ogm.transform_point_to_grid_cell(mp)
        scan_matcher.ogm.colorize_grid_black((grid_idx_i, grid_idx_j))

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
        # raise ValueError(f"\nExtracted map points do not match original map points. Extracted: {extracted_map_points}, Original: {map_points}")
        # This is expected because because we transform pure points into discrete grid cells and back
        print(f"\nExtracted map points do not match original map points.\nOriginal: {map_points}\nExtracted: {extracted_map_points}")

    #___________________________________________________________________________________________________
    # Compute measurement likelihoods 
    #___________________________________________________________________________________________________
    log_likelihoods = []
    for i, sample in enumerate(samples):
        score, log_likeli, matched_count = measurement_model.gmapping_likelihood(
            pose=sample,
            measurements=measurements,
            ogm=scan_matcher.ogm,
            usable_range=scan_matcher.max_sensor_range,
            kernel_size=1,
            fullness_threshold=scan_matcher.occ_thres,
            free_threshold=scan_matcher.occ_thres,
            free_cell_ratio=np.sqrt(2)
        )

        log_likelihoods.append(log_likeli)

    meas_probs = np.exp(log_likelihoods - np.max(log_likelihoods))

    # Normalized probs
    meas_probs /= np.sum(meas_probs)

    #___________________________________________________________________________________________________
    # Evaluation
    #___________________________________________________________________________________________________
    eval = RBPFEvaluator()
    # Compute sample errors
    if samples is not None and true_pose is not None and meas_probs is not None:
        xjs = np.asarray(samples, dtype=float)
        meas_probs = np.asarray(meas_probs, dtype=float).reshape(-1)

        if (xjs.ndim == 2 and
            xjs.shape[0] > 0 and
            xjs.shape[1] == 3 and
            meas_probs.shape[0] == xjs.shape[0]
        ):
            valid_idx = np.where(
                np.isfinite(meas_probs) & np.all(np.isfinite(xjs[:, :3]), axis=1)
            )[0]

            # Raise err if no valid indices found
            if valid_idx is None:
                raise(f"Measurement probs and xjs have no finite members in common!")

            # Extract valid values
            xjs = xjs[valid_idx]
            meas_probs = meas_probs[valid_idx]

            # Compute sample errors
            xj_pose_errors = []
            xj_trans_errors = []
            xj_rot_errors = []
            true_pose = eval._to_pose_tuple(true_pose)
            for xj in xjs:
                xj_pose = eval._to_pose_tuple(xj)
                t_err = eval.translation_error(xj_pose, true_pose)
                r_err = abs(eval.angle_diff(xj_pose[2], true_pose[2]))
                p_err = eval.pose_err(t_err, r_err, 2.0)
            
                xj_pose_errors.append(p_err)
                xj_trans_errors.append(t_err)
                xj_rot_errors.append(r_err)

            # Transform errors to numpy arrays
            xj_pose_errors = np.asarray(xj_pose_errors)
            xj_trans_errors = np.asarray(xj_trans_errors)
            xj_rot_errors = np.asarray(xj_rot_errors)

            # Compute correlation between measurement probabilities and errors            
            corr_trans, _ = spearmanr(meas_probs, -xj_trans_errors)
            corr_rot, _ = spearmanr(meas_probs, -xj_rot_errors)     
            corr_pose, _ = spearmanr(meas_probs, -xj_pose_errors)
            
            # Get min xj
            min_xj_idx = np.argmin(xj_pose_errors)
            min_xj = xjs[min_xj_idx]
            min_xj_pose_error = xj_pose_errors[min_xj_idx]
            min_xj_meas_prob = meas_probs[min_xj_idx]

            # Get xj with highest weight
            best_xj_idx = np.argmax(meas_probs)
            best_xj = xjs[best_xj_idx]
            best_xj_pose_error = xj_pose_errors[best_xj_idx]
            best_xj_meas_prob = meas_probs[best_xj_idx]
            
            # Print results
            print(f"\nCorrelation between measurement probabilities and translation errors: {corr_trans:.3f}")
            print(f"Correlation between measurement probabilities and rotation errors: {corr_rot:.3f}")
            print(f"Correlation between measurement probabilities and pose errors: {corr_pose:.3f}")

            # Print min and best xj results
            if min_xj_idx == best_xj_idx:
                print(f"\nThe sample with the min error got the highest weight.")
            
            print(f"\nXj with min error to true pose is sample {min_xj_idx} with pose: {min_xj}")
            print(f"min xj meas prob: {min_xj_meas_prob}")
            print(f"min xj pose err: {min_xj_pose_error}")
            
            print(f"\nXj with highest weight is sample {best_xj_idx} with pose: {best_xj}")
            print(f"best xj meas prob: {best_xj_meas_prob}")
            print(f"best xj pose err: {best_xj_pose_error}")

            print("\nNormalized Measurement probs:")
            for i, prob in enumerate(meas_probs):
                print(f"Sample {i}: {prob:.3f}")
        

if __name__ == "__main__":
    main()
