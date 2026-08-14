import numpy as np
import matplotlib.pyplot as plt

from math import cos, sin, radians, degrees, atan2

# Import your ICP
from .icp_scan_matching import IterativeClosestPoint

'''
Test code for tesing if icp converges to correct solution on synthetic data with known transform.

Noise can be added if wished

'''


# Noise standard deviation (m) for transformed cloud. Set to 0 for no noise.
STDV = 0.05

# Define seed for reproduceable experiments
EXPERIMENT_SEED = 42


def build_transform(tx, ty, theta_rad):
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)

    T = np.array([
        [c, -s, tx],
        [s,  c, ty],
        [0,  0,  1]
    ])

    return T


def apply_transform(points, tx, ty, theta_rad):
    """
    Apply SE(2) transform to Nx2 point cloud.
    """
    T = build_transform(tx, ty, theta_rad)

    pts_h = np.hstack([
        points,
        np.ones((points.shape[0], 1))
    ])

    transformed = (T @ pts_h.T).T

    return transformed[:, :2]


def create_test_pointcloud():
    """
    Create asymmetric pointcloud so ICP
    has unique solution.
    """

    pts = []

    # Rectangle
    for x in np.linspace(0, 4, 80):
        pts.append([x, 0])

    for y in np.linspace(0, 2, 40):
        pts.append([4, y])

    for x in np.linspace(4, 0, 80):
        pts.append([x, 2])

    for y in np.linspace(2, 0, 40):
        pts.append([0, y])

    # Add asymmetric structure
    for x in np.linspace(1, 3, 40):
        pts.append([x, 1])

    return np.array(pts)


def rotation_error_deg(theta_est, theta_gt):
    dtheta = atan2(
        sin(theta_est - theta_gt),
        cos(theta_est - theta_gt)
    )

    return degrees(abs(dtheta))



def main():

    # Use a fixed seed so repeated experiments are bitwise-identical.
    np.random.seed(EXPERIMENT_SEED)
    rng = np.random.default_rng(EXPERIMENT_SEED)

    # --------------------------------------------------
    # Create original pointcloud
    # --------------------------------------------------

    true_cloud = create_test_pointcloud()

    # --------------------------------------------------
    # Ground truth transform
    # --------------------------------------------------

    gt_tx = 2.7
    gt_ty = -0.7
    gt_theta_deg = 60.0

    gt_theta_rad = radians(gt_theta_deg)

    # --------------------------------------------------
    # Create transformed cloud
    # --------------------------------------------------

    transformed_cloud = apply_transform(
        true_cloud,
        gt_tx,
        gt_ty,
        gt_theta_rad
    )

    # Optional small noise
    noise = rng.normal(0, STDV, transformed_cloud.shape)
    transformed_cloud += noise

    # --------------------------------------------------
    # ICP Setup
    # --------------------------------------------------

    stop_params = {
        "max_iterations": 30,
        "epsilon_rel": 1e-5,
        "no_improvement_limit": 5,
        "min_error": 1e-8,
        "min_dtrans": 1e-8,
        "min_drot": 1e-8,
        "min_points": 20,
        "min_corresp": 10,
        "max_translation_jump": 10.0,
        "max_rotation_jump": np.deg2rad(180),
        "max_acceptable_mean_error": 1000.0,
    }

    icp = IterativeClosestPoint(
        stop_params=stop_params,
        max_n_points=800,
        max_correspondence_distance=2.0,
        n_neighbors=10
    )

     # --------------------------------------------------
    # IMPORTANT:
    #
    # ICP estimates transform:
    #
    # transformed_cloud -> true_cloud
    #
    # therefore GT comparison uses inverse transform
    # --------------------------------------------------

    result = icp.find_transformation(
        new_data_pointpairs=transformed_cloud,
        true_data_pointpairs=true_cloud
    )

    est_tx, est_ty, est_theta = result.transformation.flatten()

    # --------------------------------------------------
    # Ground truth inverse transform
    # --------------------------------------------------

    T_gt = build_transform(
        gt_tx,
        gt_ty,
        gt_theta_rad
    )

    T_gt_inv = np.linalg.inv(T_gt)

    gt_tx_inv = T_gt_inv[0, 2]
    gt_ty_inv = T_gt_inv[1, 2]
    gt_theta_inv = atan2(
        T_gt_inv[1, 0],
        T_gt_inv[0, 0]
    )

    # --------------------------------------------------
    # Errors
    # --------------------------------------------------

    trans_error = np.linalg.norm([
        est_tx - gt_tx_inv,
        est_ty - gt_ty_inv
    ])

    rot_error = rotation_error_deg(
        est_theta,
        gt_theta_inv
    )

    # --------------------------------------------------
    # Print Results
    # --------------------------------------------------

    print("\n==============================")
    print("GROUND TRUTH (inverse)")
    print("==============================")
    print(f"tx     = {gt_tx_inv:.6f}")
    print(f"ty     = {gt_ty_inv:.6f}")
    print(f"theta  = {degrees(gt_theta_inv):.6f} deg")

    print("\n==============================")
    print("ICP ESTIMATE")
    print("==============================")
    print(f"tx     = {est_tx:.6f}")
    print(f"ty     = {est_ty:.6f}")
    print(f"theta  = {degrees(est_theta):.6f} deg")

    print("\n==============================")
    print("ERROR")
    print("==============================")
    print(f"translation error = {trans_error:.8f} m")
    print(f"rotation error    = {rot_error:.8f} deg")

    print("\n==============================")
    print("ICP INFO")
    print("==============================")
    print(f"use transformation = {result.use_transformation}")
    print(f"reason             = {result.reason}")
    print(f"iterations         = {result.n_iterations}")
    print(f"mean error         = {result.mean_error}")

    # --------------------------------------------------
    # Visualize
    # --------------------------------------------------

    aligned_cloud = apply_transform(
        transformed_cloud,
        est_tx,
        est_ty,
        est_theta
    )

    plt.figure(figsize=(10, 10))

    plt.scatter(
        true_cloud[:, 0],
        true_cloud[:, 1],
        s=5,
        label="True Cloud"
    )

    plt.scatter(
        transformed_cloud[:, 0],
        transformed_cloud[:, 1],
        s=5,
        label="Initial Misaligned Cloud"
    )

    plt.scatter(
        aligned_cloud[:, 0],
        aligned_cloud[:, 1],
        s=5,
        label="ICP Aligned Cloud"
    )

    plt.axis("equal")
    plt.legend()
    plt.title("ICP Validation Test")

    plt.show()



if __name__ == "__main__":
    main()