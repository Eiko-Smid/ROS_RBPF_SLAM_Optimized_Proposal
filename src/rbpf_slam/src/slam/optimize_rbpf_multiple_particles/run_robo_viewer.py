from ..robo_viewer.robo_viewer_launcher import RoboViewerLauncher


START_DIR = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optm_results_mult_part/"

TRAJECTORY_COLS = {
    "true_pose_traj": (
        "true_pose_x",
        "true_pose_y",
        "true_pose_theta",
    ),
    "raw_odom_traj": (
        "raw_odom_pose_x",
        "raw_odom_pose_y",
        "raw_odom_pose_theta",
    ),
    "weighted_mean_traj": (
        "weighted_mean_pose_x",
        "weighted_mean_pose_y",
        "weighted_mean_pose_theta",
    ),
    "best_particle_traj": (
        "best_particle_pose_x",
        "best_particle_pose_y",
        "best_particle_pose_theta",
        "best_particle_weight",
    ),
    "map_traj": (
        "map_traj_x",
        "map_traj_y",
        "map_traj_theta",
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
    
