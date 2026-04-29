#!/usr/bin/env python3

import rospy
import threading
import tf2_ros

from geometry_msgs.msg import Pose, Point
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

from rbpf_slam.msg import Measurement
from rbpf_slam.msg import LogOddsMap

from dataclasses import dataclass
import time
import numpy as np

# Import classes (support both roslaunch and direct execution contexts)
try:
    from slam.scan_matcher.ogm_scan_matching import OGM
    from slam.rbpf.scan_match_factory import (
        OccupancyParams,
        SensorParams,
        MapParameter,
    )
except ModuleNotFoundError:
    from rbpf_slam.src.slam.scan_matcher.ogm_scan_matching import OGM
    from rbpf_slam.src.slam.rbpf.scan_match_factory import (
        OccupancyParams,
        SensorParams,
        MapParameter,
    )

@dataclass
class ROSParams:
    update_rate = 12
    link_state_topic = "/gazebo/link_states"
    link_state_name = "robot_vacuum_cleaner::base_link"
    base_tf_frame = "base_link"
    laser_tf_frame = "laser_scanner_link"
    scan_topic= "scan"
    map_topic= "log_odds_map"


@dataclass
class OGMParams:
    occupancy_params: OccupancyParams
    sensor_params: SensorParams
    map_param: MapParameter


def define_experiment_params():
    exp_param = OGMParams(
        occupancy_params=OccupancyParams(
            prior_probability=0.5,
            min_distance_to_border=15.0,
            increasing_probability=0.8,
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
            grid_resolution_m=0.1,
        ),  
    )

    return exp_param


def init_ogm(exp_param: OGMParams) -> OGM:
    # init OGM algorithm
    ogm = OGM(
        map_parameter=exp_param.occupancy_params.min_distance_to_border,
        occupancy_parameter= [
            exp_param.occupancy_params.prior_probability,
            exp_param.occupancy_params.increasing_probability,
            exp_param.occupancy_params.decreasing_probability,
            exp_param.occupancy_params.min_log_odds,
            exp_param.occupancy_params.max_log_odds,
        ],
        sensor_parameter= [
            exp_param.sensor_params.min_sensor_range,
            exp_param.sensor_params.max_sensor_range,
        ]
    )


    # Init empty map with predefined prior probs
    ogm.init_map(
        map_width=exp_param.map_param.map_width,
        map_height=exp_param.map_param.map_height,
        grid_resolution=exp_param.map_param.grid_resolution_m
    )

    return ogm


class OGMROSCommunication:
    def __init__(self, ogm: OGM, ros_params: ROSParams):
        # Set member variables
        self.ogm = ogm
        self.ros_params = ros_params

        self.link_state_message = None
        self.link_state_name = ros_params.link_state_name
        self.link_state_index = None
        self.laser_scan = None
        self.laser_pose_world = None

        # Cache the static base->laser transform once (2D: x, y, yaw).
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.base_to_laser_pose_2d = self.lookup_base_to_laser_transform_2d()
        
        # Define subscriber for link states and laser scan
        self.link_states_subscriber = rospy.Subscriber(
            self.ros_params.link_state_topic, LinkStates, self.link_states_callback
        )
        self.laser_scan_subscriber= rospy.Subscriber(self.ros_params.scan_topic, LaserScan, self.laser_scan_callback)

        # Define publisher for map
        self.map_publisher= rospy.Publisher(self.ros_params.map_topic, LogOddsMap, queue_size=1) 

        self.lock = threading.Lock()


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


    def link_states_callback(self, link_states: LinkStates):
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
        

    def laser_scan_callback(self, laser_scan):
        '''Receive laser scan from topic.'''
        self.lock.acquire()
        self.laser_scan= laser_scan
        self.lock.release()

    
    @staticmethod
    def transform_link_state_pose_to_planar_pose(link_state: LinkStates, link_state_index: int):
        '''
        Transforms the link state message to a planar pose, consisting of (x, y, yaw) tuple.
        '''
        link_state_pose: Pose = link_state.pose[link_state_index]

        x= link_state_pose.position.x
        y= link_state_pose.position.y
        orientation = link_state_pose.orientation
        # Transform quaternion angle's to euler angle's
        (roll, pitch, yaw)= euler_from_quaternion([orientation.x, orientation.y, orientation.z,
                                                orientation.w])
        planar_pose= (x, y, yaw)
        return planar_pose


    @staticmethod
    def transform_laser_scan_to_measurement(laser_scan: LaserScan):
        '''Transform the sensor msgs LaserScan to a list of measurement's consisting of 
        (range, bearing) tuple.'''
        min_angle= laser_scan.angle_min
        angle_increment= laser_scan.angle_increment
        bearing= min_angle
        measurements= []
        counter= 0
        # Transform LaserScan data
        for range in laser_scan.ranges:
            measurement= (range, bearing)
            bearing+= angle_increment
            measurements.append(measurement)
        return measurements    


    @staticmethod
    def transform_planar_pose(pose, pose_offset):
        '''Compose two 2D poses: world->base and base->laser => world->laser.'''
        x, y, yaw = pose
        dx, dy, dyaw = pose_offset
        transformed_x = x + np.cos(yaw) * dx - np.sin(yaw) * dy
        transformed_y = y + np.sin(yaw) * dx + np.cos(yaw) * dy
        transformed_yaw = yaw + dyaw
        return (transformed_x, transformed_y, transformed_yaw)


    def publish_occupancy_grid_message(self):
        '''Get's the logOdds map and do all necessary transformation's for publishing the Occupancy Message.'''
        # Publish current map from ogm algorithm
        self.map_publisher.publish(self.ogm.return_log_odds_map_object())


    def execute(self):
        '''Main loop for executing the algorithm.'''
        update_rate= rospy.Rate(self.ros_params.update_rate)
        while not rospy.is_shutdown():
            # rospy.loginfo("Mapping node running")
            # Check if data was received
            
            # Check if data is available
            if (
                self.link_state_message is not None
                and self.link_state_index is not None
                and self.laser_scan is not None
            ):
                rospy.loginfo_once("OGM Initalized.")

                # get data from callbacks
                with self.lock:
                    link_state = self.link_state_message
                    link_state_index = self.link_state_index 
                    laser_scan = self.laser_scan

                # Transform data
                pose = self.transform_link_state_pose_to_planar_pose(
                    link_state=link_state,
                    link_state_index=link_state_index,
                )

                # Optional world-frame laser pose for later world-point projection.
                self.laser_pose_world = self.transform_planar_pose(
                    pose,
                    self.base_to_laser_pose_2d,
                )

                measurements = self.transform_laser_scan_to_measurement(laser_scan)

                # Increase map size if necessary
                extension_needed= True
                while(extension_needed):
                    extension_needed= self.ogm.map_extension_if_necessary(self.laser_pose_world)

                # Update the map
                self.ogm.update_map(measurements, self.laser_pose_world)

                # log beam otuside map count
                if self.ogm.beam_out_map_count > 0:
                    rospy.loginfo(f"Beam outside map count: {self.ogm.beam_out_map_count}")
                
                # Transform and publish map
                self.publish_occupancy_grid_message()

            update_rate.sleep()



def main():
    # Init OGM
    exp_param = define_experiment_params()
    ogm = init_ogm(exp_param=exp_param)
    
    # Init Node
    rospy.init_node("optimized_occupancy_grid_algo_with_map_extension", anonymous=True)


    # Initialize algorithm
    ros_params = ROSParams()
    ros_ogm= OGMROSCommunication(ogm=ogm, ros_params=ros_params)

    # Start the algorithm
    ros_ogm.execute()

    

if __name__=="__main__":
    main()  