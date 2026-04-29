#!/usr/bin/env python3

import rospy
import threading

from geometry_msgs.msg import Pose, Point
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

from rbpf_slam.msg import Measurement
from rbpf_slam.msg import LogOddsMap

from dataclasses import dataclass
import time
import numpy as np

# Import classes roslaunch
# from rbpf_slam.slam.scan_matcher.ogm_scan_matching import OGM
# from rbpf_slam.slam.rbpf.scan_match_factory import (
#     OccupancyParams,
#     SensorParams,
#     MapParameter,
# )


# # Import classes programming
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
        
        # Define subscriber for link states and laser scan
        self.link_states_subscriber = rospy.Subscriber(
            self.ros_params.link_state_topic, LinkStates, self.link_states_callback
        )
        self.laser_scan_subscriber= rospy.Subscriber(self.ros_params.scan_topic, LaserScan, self.laser_scan_callback)

        # Define publisher for map
        self.map_publisher= rospy.Publisher(self.ros_params.map_topic, LogOddsMap, queue_size=1) 

        self.lock = threading.Lock()


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
            if(self.link_state_message and self.link_state_index and self.laser_scan):
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

                measurements = self.transform_laser_scan_to_measurement(laser_scan)

                # Increase map size if necessary
                extension_needed= True
                while(extension_needed):
                    extension_needed= self.ogm.map_extension_if_necessary(pose)

                # Update the map
                self.ogm.update_map(measurements, pose)
                
                # Transform and publish map
                self.publish_occupancy_grid_message()

            update_rate.sleep()



def main():
    # Init OGM
    exp_param = define_experiment_params()
    ogm = init_ogm(exp_param=exp_param)
    
    # Init Node
    rospy.init_node("optimized_occupancy_grid_algo_with_map_extension", anonymous=True)

    # Define subscriber topics
    link_state_topic = "/gazebo/link_states"
    link_state_name = "robot_vacuum_cleaner::base_link"
    scan_topic= "scan"
    map_topic= "log_odds_map"
    
    # Define update rate of mapping algorithm
    update_rate= 12                 # Highest possible rate is 15    

    # Initialize algorithm
    ros_params = ROSParams()
    ros_ogm= OGMROSCommunication(ogm=ogm, ros_params=ros_params)

    # Start the algorithm
    ros_ogm.execute()

    


if __name__=="__main__":
    main()  