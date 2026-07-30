#!/usr/bin/env python3

from typing import Tuple

from dataclasses import dataclass
from math import atan2, cos, sin, sqrt
import random
import threading
import time
import xml.etree.ElementTree as ET

import rospy
import message_filters

from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

from rbpf_slam.msg import (
    RBPFInput
)

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
    # from rbpf_slam.src.slam.optimize_rbpf.playback_defs import ExperimentParams
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
    # from slam.optimize_rbpf.playback_defs import ExperimentParams
    from slam.infrastructure.playback_recorder import PlaybackRecorder


NODE_NAME = "rbpf_data_processor_node"


@dataclass
class ROSParams:
    '''
    Stores ROS topic names and parameters required by the playback node.
    '''
    ground_truth_topic: str
    scan_topic: str
    rbpf_input_topic: str
    desired_time_window_s: float
    time_window_tolerance_s: float
    max_sync_error_s: float
    time_synchronizer_queue_size: int
    time_synchronizer_slop_s: float
    robot_start_pose: Tuple[float, float, float]
    motion_error_factor: float
    turn_error_factor: float



def load_wheel_separation():
    '''
    Load the wheel separation computed in the generated robot description.
    '''
    robot_description = rospy.get_param(
        "/robot_vacuum_cleaner_description"
    )
    robot = ET.fromstring(robot_description)
    wheel_separation_element = robot.find(
        ".//plugin[@name='differential_drive_controller']/wheelSeparation"
    )

    if (
        wheel_separation_element is None
        or wheel_separation_element.text is None
    ):
        raise RuntimeError(
            "wheelSeparation not found in robot_vacuum_cleaner_description"
        )

    wheel_separation = float(wheel_separation_element.text)
    rospy.loginfo(
        "Loaded wheel separation from robot description: %s",
        wheel_separation,
    )
    return wheel_separation


def load_ros_node_params(
    robot_start_pose: Tuple[float, float, float],
    motion_error_factor: float,
    turn_error_factor: float,
) -> ROSParams:
    '''Load fixed ROS settings and add values supplied by the launch file.'''
    config = rospy.get_param("~ros")

    try:
        topics = config["topics"]
        synchronization = config["synchronization"]

        return ROSParams(
            ground_truth_topic=topics["ground_truth"],
            scan_topic=topics["scan"],
            rbpf_input_topic=topics["rbpf_input"],
            desired_time_window_s=float(
                synchronization["desired_time_window_s"]
            ),
            time_window_tolerance_s=float(
                synchronization["time_window_tolerance_s"]
            ),
            max_sync_error_s=float(
                synchronization["max_sync_error_s"]
            ),
            time_synchronizer_queue_size=int(
                synchronization["time_synchronizer_queue_size"]
            ),
            time_synchronizer_slop_s=float(
                synchronization["time_synchronizer_slop_s"]
            ),
            robot_start_pose=robot_start_pose,
            motion_error_factor=motion_error_factor,
            turn_error_factor=turn_error_factor,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid RBPF data processor ROS configuration: {exc}"
        ) from exc



class RBPFDataProcessorNode:
    def __init__(
        self,
        ros_params: ROSParams,
        wheel_separation: float,
    ):
        # Store input parameters
        self.ros_params = ros_params
        self.wheel_separation = wheel_separation

        # Create thread locks 
        self.lock = threading.Lock()

        # Init previous synchronized pose with robot spawn pose
        self.prev_pose = tuple(self.ros_params.robot_start_pose)
        self.prev_scan_msg = None

        # Define topic the message Filter should subscribe to
        self.scan_sub = message_filters.Subscriber(
            ros_params.scan_topic,
            LaserScan,
        )

        self.ground_truth_sub = message_filters.Subscriber(
            ros_params.ground_truth_topic,
            Odometry,
        )

        # Init Time Synchronizer 
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.scan_sub, self.ground_truth_sub],
            queue_size=ros_params.time_synchronizer_queue_size,
            slop=ros_params.time_synchronizer_slop_s
        )

        # Register callback -> synced data will be send to this cb
        self.synchronizer.registerCallback(self.synchronizer_cb)

        # Add publisher
        self.rbpf_input_pub = rospy.Publisher(
            ros_params.rbpf_input_topic,
            RBPFInput,
            queue_size=10
        )

        # Define shutdown behavior
        rospy.on_shutdown(self.on_shutdown)


    def on_shutdown(self):
        '''
        Defines the shutdown behavior of the node.
        '''
        rospy.loginfo("Shutting down synchronized playback node.")


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


    @staticmethod
    def _wheelencoder_simulation(old_pose, new_pose, width, eps_alpha= 1e-3):
        '''
        Get's the pose at x_t and x_t-1, as well as robot width and computes the distance the left 
        and right wheel traveled, since the last time stamp. 

        Parameters:@
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
            motion_error_factor = self.ros_params.motion_error_factor
            turn_error_factor = self.ros_params.turn_error_factor

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


    def publish_rbpf_info(
            self,
            laser_scan: LaserScan,
            dl: float, 
            dr: float,
            pose: Tuple[float, float, float],
    ):
        # Init RBPFInput message
        msg = RBPFInput()

        msg.header = laser_scan.header
        msg.laser_scan = laser_scan

        msg.wheel_encoder.left = dl
        msg.wheel_encoder.right = dr
        msg.true_pose.x = pose[0]
        msg.true_pose.y = pose[1]
        msg.true_pose.theta = pose[2]

        # Publish data 
        self.rbpf_input_pub.publish(msg)


    def synchronizer_cb(
            self, 
            laser_scan: LaserScan,
            ground_truth_odom: Odometry,
        ) -> None:
            '''
            Callback functions that readds the synchronized laser scan and ground truth odometry messages, processes them
            and publishes the RBPF input message. 
            The method simulates the wheel encoder data based on the received ground truth odometry and publishes it together
            with the laser scan data, while ensuring the data is within time thresholds.

            Parameters
            ----------
            laser_scan: LaserScan
                The synchronized laser scan message.
            ground_truth_odom: Odometry
                The synchronized ground truth odometry message.
            
            '''
            # Read synchronized data 
            with self.lock:
                # Check if this is the first scan message received, if so, store it and return
                if self.prev_scan_msg is None:
                    # Init message
                    self.prev_scan_msg: LaserScan = laser_scan
                    return            
                else:
                    # Compute time difference
                    new_laser_time = laser_scan.header.stamp
                    old_laser_time = self.prev_scan_msg.header.stamp
                    scan_time_diff = (new_laser_time - old_laser_time).to_sec()
    
                # Copy data -> glob var are free now -> leave lock
                laser_scan_cp = laser_scan
                ground_truth_odom_cp = ground_truth_odom
    
            # Accept new data pair if time difference is within thres
            if scan_time_diff > (
                self.ros_params.desired_time_window_s
                - self.ros_params.time_window_tolerance_s
            ):
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
                dl, dr = self._wheelencoder_simulation(
                    old_pose=self.prev_pose,
                    new_pose=pose,
                    width=self.wheel_separation,
                )
    
                # Add noise to wheel encoder data
                dl, dr = self.add_wheel_encoder_noise(
                    left_control=dl,
                    right_control=dr,
                )
                
                # Update prev data
                self.prev_scan_msg = laser_scan_cp
                self.prev_pose = pose

                # Publish rbpf input data
                self.publish_rbpf_info(
                    laser_scan=laser_scan_cp,
                    dl=dl, 
                    dr=dr,
                    pose=pose
                )
            return       


    def exe(self):
        rospy.spin()




def main():
    '''Initializes parameters and starts the synchronized playback node.'''
    # Init node
    rospy.init_node(NODE_NAME)
    config_name = rospy.get_param("~config_name")

    # Get motion error parameters
    motion_error_factor = float(
        rospy.get_param("/motion_error_factor")
    )
    turn_error_factor = float(
        rospy.get_param("/turn_error_factor")
    )

    # Get robot spawn pose
    robot_start_pose = (
        float(rospy.get_param("/spawn_x")),
        float(rospy.get_param("/spawn_y")),
        float(rospy.get_param("/spawn_yaw")),
    )

    # Load wheel separation from the generated robot description
    wheel_separation = load_wheel_separation()
    ros_params = load_ros_node_params(
        robot_start_pose=robot_start_pose,
        motion_error_factor=motion_error_factor,
        turn_error_factor=turn_error_factor,
    )

    # Display parameters 
    rospy.loginfo("Loaded RBPF data processor configuration: %s", config_name)
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

    # RUn RBPF data processor
    rbpf_data_processor = RBPFDataProcessorNode(
        ros_params=ros_params,
        wheel_separation=wheel_separation
    )

    rbpf_data_processor.exe()



if __name__ == "__main__":
    main()
