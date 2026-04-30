#!/usr/bin/env python3

# debugpy.listen(("0.0.0.0", 5678))
# print("⏳ Waiting for debugger attach...")
# debugpy.wait_for_client()
# print("✅ Debugger attached")

from typing import List, Tuple

import rospy
import threading
import tf2_ros

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, Point, Quaternion
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from rbpf_slam.msg import Measurement
from rbpf_slam.msg import LogOddsMap

from dataclasses import dataclass
import time
import numpy as np


NODE_NAME = "rbpf_slam_node"

try: 
    from rbpf_slam.src.slam.rbpf.rbpf import (
        RBPFFactory,
        ParticleParams,
        MotionModelParams,
        MeasurementModelParams,
        RBPF
    )
    from rbpf_slam.src.slam.rbpf.scan_match_factory import (
        OccupancyParams,
        SensorParams,
        MapParameter,
        ICPParams,
        RobotParams,
        ScanMatcherParams,
        ScanMatchFactory,
    )
    from rbpf_slam.src.slam.optimize_rbpf.playback_defs import ExperimentParams, PlaybackData
    
except ModuleNotFoundError:
    from slam.rbpf.rbpf import (
        RBPFFactory,
        ParticleParams,
        MotionModelParams,
        MeasurementModelParams,
        RBPF
    )

    from slam.rbpf.scan_match_factory import (
        OccupancyParams,
        SensorParams,
        MapParameter,
        ICPParams,
        RobotParams,
        ScanMatcherParams,
        ScanMatchFactory,
    )

    from slam.optimize_rbpf.playback_defs import ExperimentParams, PlaybackData

    
from rbpf_slam.msg import WheelEncoder
from rbpf_slam.msg import PoseErr2D


'''
What we need

Topics
- laser_scan
- gazebo link states
- odom
- map


'''


@dataclass
class ROSParams:
    # Update rate
    update_rate = 2
    
    # Link states topic and params
    link_state_topic = "/gazebo/link_states"
    link_state_name = "robot_vacuum_cleaner::base_link"
    
    # scan topic
    scan_topic= "scan"
    # log odds map topic
    map_topic= "log_odds_map"
    # odom topic
    wheel_encoder_topic= "wheel_encoder"

    # TFs
    base_tf_frame = "base_link"
    laser_tf_frame = "laser_scanner_link"



def define_exp_parameter() -> ExperimentParams:
    # Compute wheel separation
    h_chassis= 0.15
    dist_chassis_to_ground= h_chassis/5
    r_wheel= h_chassis/2 + dist_chassis_to_ground
    w_wheel= 0.3 * r_wheel
    r_chassis= 0.25
    wheel_separation= 2 * r_chassis + w_wheel

    exp_param = ExperimentParams(
        occupancy_params=OccupancyParams(
                    prior_probability=0.5,
                    min_distance_to_border=10.0,
                    increasing_probability=0.7,
                    decreasing_probability=0.35,
                    min_log_odds=-5.0,
                    max_log_odds=5.0,
                ),
                sensor_params=SensorParams(
                    min_sensor_range=0.1,
                    max_sensor_range=10.0,
                ),
                map_param=MapParameter(
                    map_width=10.0,
                    map_height=10.0,
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
                every_nth_scan=1,
                proposal_sigma_xy=0.1,
                proposal_sigma_theta=0.05,
                proposal_n_samples=10,
                tag=("First rbpf node run"),
            )

    return exp_param


class RBPFROS:
    def __init__(self, rbpf: RBPF, ros_params: ROSParams):
        # Init members
        self.rbpf: RBPF = rbpf
        self.ros_params = ros_params

        self.distance_left_wheel = 0.0
        self.distance_right_wheel = 0.0

        # Defien link states
        self.link_state_message = None
        self.link_state_name = ros_params.link_state_name
        self.link_state_index = None
        self.laser_scan = None
        self.laser_pose_world = None

        # Define topics
        self.link_state_sub = rospy.Subscriber(
            name=self.ros_params.link_state_topic,
            data_class=LinkStates,
            callback=self.link_states_cb,
        )

        # Init laser scan sub
        self.laser_scan_sub = rospy.Subscriber(
            name=self.ros_params.scan_topic,
            data_class=LaserScan,
            callback=self.laser_scan_cb,
        )

        # Init wheel encoder sub
        self.wheel_encode_sub = rospy.Subscriber(
            name=self.ros_params.wheel_encoder_topic,
            data_class=WheelEncoder,
            callback=self.wheel_encoder_cb,
        )

        # Define publisher
        self.map_publisher = rospy.Publisher(
            name=self.ros_params.map_topic,
            data_class=LogOddsMap,
            queue_size=1
        )

        # Defien thread lcoker obj
        self.lock = threading.Lock()


        # Cache the static base->laser transform (x, , theta)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.base_to_laser_pose_2d = self.lookup_base_to_laser_transform_2d()
        

    def lookup_base_to_laser_transform_2d(self):
        '''Look up static transform from base frame to laser frame once.'''
        while not rospy.is_shutdown():
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.ros_params.base_tf_frame,
                    self.ros_params.laser_tf_frame,
                    rospy.Time(0),
                    rospy.Duration(1.0),
                )
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                (_, _, yaw) = euler_from_quaternion(
                    [rotation.x, rotation.y, rotation.z, rotation.w]
                )
                rospy.loginfo(
                    "Cached base->laser offset (2D): x=%.3f, y=%.3f, yaw=%.3f rad",
                    translation.x,
                    translation.y,
                    yaw,
                )
                return (translation.x, translation.y, yaw)
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                rospy.logwarn_throttle(
                    5.0,
                    f"Waiting for TF {self.ros_params.base_tf_frame} -> {self.ros_params.laser_tf_frame}",
                )

        return (0.0, 0.0, 0.0)


    def on_shutdown(self):
        '''
        Shutdown callback function to log time jumps and store python playback data if enabled.
        '''
        pass


    def link_states_cb(self, link_states: LinkStates):
        '''Receive gazebo link state from topic.'''
        with self.lock:
            # Extract message
            self.link_state_message = link_states

            # Find link state name index -> base_link index
            if self.link_state_index is None:
                try:
                    self.link_state_index = link_states.name.index(self.link_state_name)
                    rospy.loginfo(f"Found link state index: {self.link_state_index}")
                except ValueError:
                    rospy.logwarn_throttle(5.0, f"Link {self.link_state_name} not found in Gazebo link states.")

            if self.link_state_index is None:
                for i in range(len(link_states.name)):
                    if self.link_state_name == link_states.name[i]:
                        self.link_state_index = i
                        break
    

    def laser_scan_cb(self, laser_scan):
        '''Receive laser scan from topic.'''
        self.lock.acquire()
        self.laser_scan= laser_scan
        self.lock.release()


    def wheel_encoder_cb(self, distance):
        '''Accumulate the distances of the left and right wheel.'''
        self.lock.acquire()
        self.distance_left_wheel+= distance.left
        self.distance_right_wheel+= distance.right
        self.lock.release()

    

    def publish_occupancy_grid_message(self):
        '''Get's the logOdds map and do all necessary transformation's for publishing the Occupancy Message.'''
        # Publish current map from ogm algorithm
        # TODO: Publish best particle map here !
        self.map_publisher.publish(self.rbpf.particles[0].scan_matcher.ogm.return_log_odds_map_object())


    def publish_pose(self, topic_key: str, pose: Tuple[float, float, float]):
        '''
        Publishes the given pose as a Pose message to the topic corresponding to the topic key. 
        Parameters:
        -----------
        topic_key: str
            The key of the topic the messages should be published to.
        pose: Tuple[float, float, float]
            The pose to be published, given as a tuple of (x, y, yaw).
        '''
        msg = Pose()
        # Define position
        msg.position.x = pose[0]
        msg.position.y = pose[1]
        msg.position.z = 0.0

        # Define angle
        roll, pitch, yaw, w = quaternion_from_euler(0, 0, pose[2])
        orientation = Quaternion(x=roll, y=pitch, z=yaw)
        msg.orientation = orientation

        # Publish message
        self.publishers[topic_key].publish(msg)


    @staticmethod
    def compute_pose_err(true_pose, unaccurate_pose):
        '''
        Computes the error between the true pose and the uncertain pose, reported by the scan matcher.

        Returns:
        --------
        orientation_error: float
            The orientation error in radians.
        orientation_error_grad: float
            The orientation error in degree, which is more intuitive to interpret.
        '''
        x_true, y_true, yaw_true = true_pose
        x_uncertain, y_uncertain, yaw_uncertain = unaccurate_pose

        # Compute position error
        position_error = np.sqrt((x_true - x_uncertain) ** 2 + (y_true - y_uncertain) ** 2)

        # Compute orientation error
        orientation_error = yaw_true - yaw_uncertain
        orientation_error = np.atan2(np.sin(orientation_error), np.cos(orientation_error))

        orientation_error_grad = orientation_error * 180 / np.pi

        return position_error, orientation_error, orientation_error_grad


    def execute(self):
        update_rate = rospy.Rate(self.ros_params.update_rate)

        while not rospy.is_shutdown():
            self.publish_occupancy_grid_message()
            update_rate.sleep()




def main():
    # Init OGM
    exp_param = define_exp_parameter()

    # Init rbpf
    rbpf = RBPFFactory().create(
            scan_match_fac=ScanMatchFactory(),
            particle_params=exp_param.particle_params,
            occ_param=exp_param.occupancy_params,
            sens_params=exp_param.sensor_params,
            map_param=exp_param.map_param,
            icp_params=exp_param.icp_params,
            robot_params=exp_param.robot_params,
            scan_matcher_params=exp_param.scan_matcher_params,
            motion_model_params=exp_param.motion_model_params,
            measurement_model_params=exp_param.measurement_model_params,
        ) 
   
    
    # Init Node
    rospy.init_node(NODE_NAME)


    # Initialize algorithm
    ros_params = ROSParams()
    
    ros_rbpf = RBPFROS(
        rbpf=rbpf,
        ros_params=ros_params,
    )

    # Start the algorithm
    ros_rbpf.execute()

    

if __name__=="__main__":
    main()  