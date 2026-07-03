#!/usr/bin/env python3

from typing import Tuple, Optional

from dataclasses import dataclass
from math import atan2, cos, sin, sqrt
import random
import threading
import time

import rospy
import message_filters

from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

try:
    from rbpf_slam.src.slam.rbpf.rbpf import (
        ParticleParams,
        MotionModelParams,
        MeasurementModelParams,
    )
    from rbpf_slam.src.slam.rbpf.scan_match_factory import (
        OccupancyParams,
        SensorParams,
        MapParameter,
        ICPParams,
        RobotParams,
        ScanMatcherParams,
    )
    from rbpf_slam.src.slam.optimize_rbpf.playback_defs import ExperimentParams
    from rbpf_slam.src.slam.infrastructure.playback_recorder import PlaybackRecorder

except ModuleNotFoundError:
    from slam.rbpf.rbpf import (
        ParticleParams,
        MotionModelParams,
        MeasurementModelParams,
    )
    from slam.rbpf.scan_match_factory import (
        OccupancyParams,
        SensorParams,
        MapParameter,
        ICPParams,
        RobotParams,
        ScanMatcherParams,
    )
    from slam.optimize_rbpf.playback_defs import ExperimentParams
    from slam.infrastructure.playback_recorder import PlaybackRecorder


TAG = (
    "AWS small house map on different path with synced playback."
)

MAP_NAME = "AWS_Robot_Maker_Small_House_alt_path"
NODE_NAME = "playback_node"
PLAYBACK_DIR = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/python_playback/"


@dataclass
class ROSParams:
    '''
    Stores ROS topic names and parameters required by the playback node.

    The node processes every nth laser scan. With a laser update rate of 10 Hz
    and record_every_nth_scan=5, one playback step is recorded every 0.5 s.
    '''
    # Ground-truth odometry topic published by libgazebo_ros_p3d.so
    ground_truth_topic: str = "/ground_truth/odom"

    # Scan topic
    scan_topic: str = "scan"

    # Laser scanner
    # Desired time window between 2 scans [s]
    des_time_window = 0.5 
    # If time diff is > des_time_window + d_time_window we take this scan 
    d_time_window = 0.1 * des_time_window

    # Reject scan/ground-truth pairs with a larger interpolation gap [s]
    # This threshold is checked for both bracketing poses separately.
    max_sync_error_s: float = 0.02

    # Number of ground-truth poses and selected scans stored temporarily
    # Que size for time syncrhomizer (stores that many values at max to find matching pair)
    time_synchronizer_queue_size = 50
    # Time difference between the topic data that is accepted for valid pairs [s]
    time_synchronizer_slop = 0.02

    # Robot spawn pose used as old pose for the first recorded control
    robot_start_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Artificial wheel encoder noise factors
    motion_error_factor: Optional[float] = None
    turn_error_factor: Optional[float] = None

    # Laser noise metadata
    laser_range_resolution: Optional[float] = None
    laser_noise_type: Optional[str] = None
    laser_noise_mean: Optional[float] = None
    laser_noise_stddv: Optional[float] = None


def define_exp_parameter() -> ExperimentParams:
    '''
    Defines the experiment parameters stored together with the playback data.

    Returns
    -------
    ExperimentParams
        Parameters used by the RBPF, map, sensor and scan matcher.
    '''
    # Compute wheel separation
    h_chassis = 0.15
    dist_chassis_to_ground = h_chassis / 5
    r_wheel = h_chassis / 2 + dist_chassis_to_ground
    w_wheel = 0.3 * r_wheel
    r_chassis = 0.25
    wheel_separation = 2 * r_chassis + w_wheel

    exp_param = ExperimentParams(
        occupancy_params=OccupancyParams(
            prior_probability=0.5,
            min_distance_to_border=13.0,
            increasing_probability=0.7,
            decreasing_probability=0.3,
            min_log_odds=-5.0,
            max_log_odds=5.0,
        ),
        sensor_params=SensorParams(
            min_sensor_range=0.1,
            max_sensor_range=10.0,
        ),
        map_param=MapParameter(
            map_width=25.0,
            map_height=25.0,
            grid_resolution_m=0.05,
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
        ),
        robot_params=RobotParams(
            wheel_separation=wheel_separation,
        ),
        scan_matcher_params=ScanMatcherParams(
            occ_thres=1.2,
            delta_r=0.6,
        ),
        particle_params=ParticleParams(
            n_particles=40,
            start_pose=(0.0, 0.0, 0.0),
        ),
        motion_model_params=MotionModelParams(
            sigma_x=0.2,
            sigma_y=0.2,
            sigma_theta=0.15,
            wheel_separation=wheel_separation,
            ctrl_motion_fac=0.1,
            ctrl_turn_fac=0.20,
        ),
        measurement_model_params=MeasurementModelParams(
            sigma_measurement=0.2,
        ),
        every_nth_scan_filter=4,
        every_nth_scan_map=2,
        proposal_sigma_xy=0.1,
        proposal_sigma_theta=0.05,
        proposal_n_samples=10,
        tag=TAG,
    )

    return exp_param


@dataclass
class RECORDParams:
    '''Stores parameters used by the PlaybackRecorder.'''
    enable_recording: bool = True
    output_dir: str = PLAYBACK_DIR


def build_metadata(exp_params: ExperimentParams, ros_params: ROSParams):
    '''
    Builds the metadata dictionary stored together with the playback data.
    '''
    return {
        "map": MAP_NAME,

        "robot_start_pose": ros_params.robot_start_pose,

        "sensor_range_max": exp_params.sensor_params.max_sensor_range,
        "sensor_range_min": exp_params.sensor_params.min_sensor_range,
        "laser_range_resolution": ros_params.laser_range_resolution,
        "laser_noise_type": ros_params.laser_noise_type,
        "laser_noise_mean": ros_params.laser_noise_mean,
        "laser_noise_stddv": ros_params.laser_noise_stddv,

        "wheel_separation": exp_params.robot_params.wheel_separation,
        "wheel_encoder_sim_motion_error_factor": ros_params.motion_error_factor,
        "wheel_encoder_sim_turn_error_factor": ros_params.turn_error_factor,

        # Time synchronization parameters
        "time_synchronizer_queue_size": ros_params.time_synchronizer_queue_size,
        "time_synchronizer_slop": ros_params.time_synchronizer_slop,

        "ground_truth_topic": ros_params.ground_truth_topic,
        "scan_topic": ros_params.scan_topic,

        "data_time_seperation_s": ros_params.des_time_window,
        "d_time_window": ros_params.d_time_window,
        "max_sync_error_s": ros_params.max_sync_error_s,

        "n_particles": exp_params.particle_params.n_particles,
        "comment": exp_params.tag,
    }


class ROSPlaybackNode:
    '''
    Records synchronized playback data for offline RBPF SLAM evaluation.

    Processing pipeline
    -------------------
    1. Receive every laser scan and select every nth scan.
    2. Wait until timestamped ground-truth poses passed the scan timestamp.
    3. Interpolate the ground-truth pose to the scan timestamp.
    4. Compute left/right wheel travel from the previous synchronized pose.
    5. Add artificial wheel encoder noise.
    6. Store scan, controls and synchronized true pose as one playback step.
    '''

    def __init__(
        self,
        ros_params: ROSParams,
        record_params: RECORDParams,
        exp_param: ExperimentParams,
    ):
        # Store members
        self.record_params = record_params
        self.ros_params = ros_params
        self.exp_params = exp_param

        # Create thread locks 
        self.lock = threading.Lock()

        # Build metadata
        metadata = build_metadata(
            exp_params=exp_param,
            ros_params=ros_params,
        )

        # Init recorder
        self.recorder = PlaybackRecorder(
            output_dir=record_params.output_dir,
            metadata=metadata,
        )
        
        # Init previous synchronized pose with robot spawn pose
        self.prev_pose = tuple(self.ros_params.robot_start_pose)
        self.prev_scan_msg = None

        # Define topic the message Filter should subscribe to
        self.scan_sub = message_filters.Subscriber(
            "/scan",
            LaserScan,
        )

        self.ground_truth_sub = message_filters.Subscriber(
            "/ground_truth/odom",
            Odometry,
        )

        # Init Time Synchronizer 
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.scan_sub, self.ground_truth_sub],
            queue_size=ros_params.time_synchronizer_queue_size,
            slop=ros_params.time_synchronizer_slop
        )

        # Register callback -> synced data will be send to this cb
        self.synchronizer.registerCallback(self.synchronizer_cb)

        # Define shutdown behavior
        rospy.on_shutdown(self.on_shutdown)


    def on_shutdown(self):
        '''Defines the shutdown behavior of the node.'''
        rospy.loginfo("Shutting down synchronized playback node.")

    
    def synchronizer_cb(
            self, 
            laser_scan: LaserScan,
            ground_truth_odom: Odometry,
    ):
        # Validate if laser scan timestamp is within window
        with self.lock:
            if self.prev_scan_msg is None:
                # Init message
                self.prev_scan_msg: LaserScan = laser_scan
                return            
            else:
                # Compute time difference
                new_laser_time = laser_scan.header.stamp
                old_laser_time = self.prev_scan_msg.header.stamp
                time_diff = (new_laser_time - old_laser_time).to_sec()

            # Copy data -> glob var are free now -> leave lock
            laser_scan_cp = laser_scan
            ground_truth_odom_cp = ground_truth_odom

        # Accept new data pair
        if time_diff > (self.ros_params.des_time_window - self.ros_params.d_time_window):
            # Compute time difference between scan and ground truth odom
            dt_scan_ground_truth = abs(laser_scan_cp.header.stamp.to_sec() - ground_truth_odom_cp.header.stamp.to_sec())
            # rospy.loginfo(                
            #     f"time_diff_scan_ground_truth={dt_scan_ground_truth * 1000.0:.2f} ms"
            # )

            # Check if synchronization error is within threshold, otherwise skip 
            if dt_scan_ground_truth > self.ros_params.max_sync_error_s:
                rospy.logwarn(
                    f"Skipping pair: synchronization error "
                    f"{dt_scan_ground_truth * 1000.0:.2f} ms"
                )
                return

            # Extract ground truth pose 
            pose = self.transform_pose_to_planar_pose(pose=ground_truth_odom_cp.pose.pose)
            # rospy.loginfo(f"True pose: x={pose[0]:.2f}, y={pose[1]:.2f}, yaw={pose[2]:.2f}")
            
            # Simulate wheel encoder data 
            dl, dr = self.wheelencoder_simulation(
                old_pose=self.prev_pose,
                new_pose=pose,
                width=self.exp_params.robot_params.wheel_separation
            )

            # Add noise to wheel encoder data
            dl, dr = self.add_wheel_encoder_noise(
                left_control=dl,
                right_control=dr,
            )

            # Store data
            self.recorder.record_step(
                t=time.perf_counter(),
                t_ros=laser_scan_cp.header.stamp.to_sec(),
                dl=dl,
                dr=dr,
                true_pose=pose, 
                laser_scan=laser_scan_cp,
            )
            
            # Update prev data
            self.prev_scan_msg = laser_scan_cp
            self.prev_pose = pose
        return
        


    @staticmethod
    def wheelencoder_simulation(old_pose, new_pose, width, eps_alpha= 1e-3):
        '''
        Get's the pose at x_t and x_t-1, as well as robot width and computes the distance the left 
        and right wheel traveled, since the last time stamp. 

        Parameters:
        ----------
        old_pose: tuple
            The pose at time x_t-1, given as (x, y, theta)
        new_pose: tuple
            The pose at time x_t, given as (x, y, theta)
        width: float
            The width of the robot, given as distance between the two wheels
        eps_alpha: float
            Threshold to determine if a turn took place, given as minimum angle in radians

        Returns:
        -------
        left_control: float
            The distance the left wheel traveled since the last time stamp
        right_control: float
            The distance the right wheel traveled since the last time stamp
        '''
        old_x, old_y, old_theta= old_pose
        new_x, new_y, new_theta= new_pose
        # Calculate alpha (turning angle)
        alpha= new_theta - old_theta
        alpha= atan2(sin(alpha), cos(alpha))

        # Compute direct distance between x_t and x_t-1
        dist = sqrt((new_x-old_x)**2 + (new_y - old_y)**2)

        # If turning took place
        if(abs(alpha) > eps_alpha):
            # Calculate turning radius 
            radius = dist / (2 * sin(alpha/2))
            # Calculate left and right control
            width_by_two= width / 2
            left_control= (radius - width_by_two) * alpha
            right_control= (radius + width_by_two) * alpha
        else:
            # If not turning took place
            left_control= dist
            right_control= dist
        return (left_control, right_control)


    def add_wheel_encoder_noise(
        self,
        left_control: float,
        right_control: float,
    ) -> Tuple[float, float]:
        '''
        Adds Gaussian motion- and turn-dependent noise to wheel travel values.

        The variance follows the same model as the previous wheel encoder node:
        motion noise depends on each wheel distance and turn noise depends on the
        difference between left and right wheel travel.
        '''
        motion_error_factor = float(self.ros_params.motion_error_factor or 0.0)
        turn_error_factor = float(self.ros_params.turn_error_factor or 0.0)

        # Compute variance contributions
        control_difference = left_control - right_control
        turn_variance = (
            turn_error_factor * control_difference
        ) ** 2

        left_variance = (
            motion_error_factor * left_control
        ) ** 2 + turn_variance
        right_variance = (
            motion_error_factor * right_control
        ) ** 2 + turn_variance

        left_standard_deviation = sqrt(left_variance)
        right_standard_deviation = sqrt(right_variance)

        # Add zero-mean Gaussian noise around ideal wheel travel
        noisy_left_control = random.gauss(
            left_control,
            left_standard_deviation,
        )
        noisy_right_control = random.gauss(
            right_control,
            right_standard_deviation,
        )

        return noisy_left_control, noisy_right_control


    @staticmethod
    def transform_pose_to_planar_pose(
        pose: Pose
    ) -> Tuple[float, float, float]:
        '''
        Transforms the pose message to a planar pose, consisting of (x, y, yaw) tuple.
        '''
        x= pose.position.x
        y= pose.position.y
        orientation = pose.orientation
        # Transform quaternion angle's to euler angle's
        (roll, pitch, yaw)= euler_from_quaternion([orientation.x, orientation.y, orientation.z,
                                                orientation.w])
        planar_pose= (x, y, yaw)
        return planar_pose
    

    def exe(self):
        '''Keeps the callback-driven playback node alive.'''
        rospy.spin()



def main():
    '''Initializes parameters and starts the synchronized playback node.'''
    # Init node
    rospy.init_node(NODE_NAME)

    # Get motion error parameters
    motion_error_factor = rospy.get_param(
        "/motion_error_factor"
    )
    turn_error_factor = rospy.get_param(
        "/turn_error_factor"
    )

    # Get robot spawn pose
    spawn_x = rospy.get_param("/spawn_x")
    spawn_y = rospy.get_param("/spawn_y")
    spawn_yaw = rospy.get_param("/spawn_yaw")
    robot_start_pose = (spawn_x, spawn_y, spawn_yaw)

    # Get laser noise parameters
    laser_range_resolution = rospy.get_param("/laser_range_resolution")
    laser_noise_type = rospy.get_param("/laser_noise_type")
    laser_noise_mean = rospy.get_param("/laser_noise_mean")
    laser_noise_stddv = rospy.get_param("/laser_noise_stddv")

    # Define experiment and recording parameters
    exp_param = define_exp_parameter()
    rec_params = RECORDParams()
    ros_params = ROSParams()

    # Set runtime ROS parameters
    ros_params.robot_start_pose = robot_start_pose
    ros_params.motion_error_factor = motion_error_factor
    ros_params.turn_error_factor = turn_error_factor

    ros_params.laser_range_resolution = laser_range_resolution
    ros_params.laser_noise_type = laser_noise_type
    ros_params.laser_noise_mean = laser_noise_mean
    ros_params.laser_noise_stddv = laser_noise_stddv

    # Display parameters 
    rospy.loginfo(f"Node {NODE_NAME} started with parameters:")
    rospy.loginfo(
        "Robot start pose: x={:.2f}, y={:.2f}, yaw={:.2f}".format(
            *robot_start_pose
        )
    )
    rospy.loginfo(f"Ground-truth topic: {ros_params.ground_truth_topic}")
    rospy.loginfo(f"Motion error factor: {motion_error_factor}")
    rospy.loginfo(f"Turn error factor: {turn_error_factor}")
    
    rospy.loginfo(
        f"Maximum synchronization error: "
        f"{ros_params.max_sync_error_s * 1000.0:.1f} ms"
    )
    rospy.loginfo(f"Laser range resolution: {laser_range_resolution}")
    rospy.loginfo(f"Laser noise type: {laser_noise_type}")
    rospy.loginfo(f"Laser noise mean: {laser_noise_mean}")
    rospy.loginfo(f"Laser noise stddv: {laser_noise_stddv}")

    # Init playback node
    playback_node = ROSPlaybackNode(
        ros_params=ros_params,
        record_params=rec_params,
        exp_param=exp_param,
    )

    # Run node
    playback_node.exe()


if __name__ == "__main__":
    main()
