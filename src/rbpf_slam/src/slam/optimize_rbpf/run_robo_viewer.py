from ..robo_viewer.robo_viewer_launcher import RoboViewerLauncher


START_DIR = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optimization_results/"

TRAJECTORY_COLS = {
    "true_pose_traj": (
        "true_pose_x",
        "true_pose_y",
        "true_pose_theta_deg",        
    ),
    "raw_odom_traj": (
        "raw_odom_pose_x",
        "raw_odom_pose_y",
        "raw_odom_pose_theta_deg",
    ),
    "scan_match_traj": (
        "scan_match_pose_x",
        "scan_match_pose_y",
        "scan_match_pose_theta_deg",
    ),
    "est_trajectory": (
        "est_pose_x",
        "est_pose_y",
        "est_pose_theta_deg",
    ),
}


def main():
    # Init robo viewer
    robo_viewer_launcher = RoboViewerLauncher(
        trajectory_columns=TRAJECTORY_COLS,
        start_directory=START_DIR
    )

    # Launch
    robo_viewer_launcher.run()


if __name__ == "__main__":
    main()
    
