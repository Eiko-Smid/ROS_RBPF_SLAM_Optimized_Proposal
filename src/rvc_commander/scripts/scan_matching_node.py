#!/usr/bin/env python3

# Init debugger
# import debugpy
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger...")
# debugpy.wait_for_client()

import os
import sys
import csv
from typing import List, Tuple
import time

from attr import dataclass
import rospy
import threading

# For math
import numpy as np
from math import sin, cos, pi, atan2, sqrt, isfinite

# TFs
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Quaternion

# Ros msgs
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import LaserScan
from nav_msgs.srv import GetMap

# Prefer source modules in this scripts directory over catkin wrapper scripts in devel/lib.
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# if SCRIPT_DIR not in sys.path:
#     sys.path.insert(0, SCRIPT_DIR)

# Own messages
from rvc_commander.msg import Measurement
from rvc_commander.msg import LogOddsMap
from rvc_commander.msg import WheelEncoder
from rvc_commander.msg import PoseErr2D

# Import Scan matching classes roslaunch
from rvc_commander.slam.icp_scan_matching import IterativeClosestPoint
from rvc_commander.slam.ogm_scan_matching import OGM
from rvc_commander.slam.scan_matcher import ScanMatcher

# Import Scan matching classes programming
# from rvc_commander.src.rvc_commander.slam.icp_scan_matching import IterativeClosestPoint
# from rvc_commander.src.rvc_commander.slam.ogm_scan_matching import OGM
# from rvc_commander.src.rvc_commander.slam.scan_matcher import ScanMatcher


'''
Description

ICP scan matching node. 

'''

@dataclass
class PoseError:
    translation: float = 0.0
    rotation: float = 0.0


class ScanMatchingNode:
    TRUE_POSE = "true_pose"
    PRED_POSE = "pred_pose"
    SCAN_MATCH_POSE = "scan_match_pose"
    POSE_ERR_TRUE_PRED = "pose_err_true_pred"
    POSE_ERR_TRUE_SCAN_MATCH = "pose_err_true_scan_match"

    def __init__(
            self,
            ros_parameter: Tuple[str, str, str, str, float],
            scan_matcher: ScanMatcher,
            every_nth_ray: int,
            headers_to_write: List[str],
            storage_filename: str = "scan_matching_data.csv",
            storage_dir: str = "/home/smide/work/ros_workspaces/ros_ws/src/rvc_commander/data/",
    ) -> None:
        # Extract ros parameter
        link_state_name, link_state_topic, scan_topic, wheel_encoder_topic, update_rate = ros_parameter

        self.update_rate = update_rate
        self.storage_path = storage_dir + str(int(time.time())) + "_" + storage_filename

        self.headers_to_write = headers_to_write

        # Define scan matcher member
        self.scan_matcher = scan_matcher

        # initialize poses
        self.true_pose = self.scan_matcher.get_pose()    # The true pose of the robot, extracted from the gazebo link state message
        self.scan_match_pose = self.true_pose
        self.predicted_pose = self.true_pose

        # init pose errors
        # Pose err between true pose and predicted pose
        self.pose_err_true_pred = PoseError()
        # Pose err between true pose and scan match pose
        self.pose_err_true_scan_match = PoseError()

        # Init members 
        self.link_state_message = None
        self.laser_scan = None
        self.distance_left_wheel = 0.0
        self.distance_right_wheel = 0.0
        self.every_nth_ray = every_nth_ray
        self.min_valid_measurements = 3

        self.lock = threading.Lock()

        # Init link state sub
        self.link_state_message = None
        self.link_state_name = link_state_name
        self.link_state_index = None
        self.link_state_sub = rospy.Subscriber(
            name=link_state_topic,
            data_class=LinkStates,
            callback=self.link_state_cb,
        )

        # Init laser scan sub
        self.laser_scan_sub = rospy.Subscriber(
            name=scan_topic,
            data_class=LaserScan,
            callback=self.laser_scan_cb,
        )

        # Init wheel encoder sub
        self.wheel_encode_sub = rospy.Subscriber(
            name=wheel_encoder_topic,
            data_class=WheelEncoder,
            callback=self.wheel_encoder_cb,
        )

        # Define pose publisher
        self.publishers = {
            self.TRUE_POSE: rospy.Publisher("true_pose", Pose, queue_size=5),
            self.PRED_POSE: rospy.Publisher("pred_pose", Pose, queue_size=5),
            self.SCAN_MATCH_POSE: rospy.Publisher("scan_match_pose", Pose, queue_size=5),
            self.POSE_ERR_TRUE_PRED: rospy.Publisher("pose_err_true_pred", PoseErr2D, queue_size=5),
            self.POSE_ERR_TRUE_SCAN_MATCH: rospy.Publisher("pose_err_true_scan_match", PoseErr2D, queue_size=5),
        }


    def get_info(self) -> dict:
        '''
        Returns a dictionary containing the current state of the scan matching node. This includes the current true pose, 
        the current predicted pose, the current scan matching pose and the current state of the ICP stop condition.
        '''
        info = self.scan_matcher.get_info()

        info["true_pose"] = self.true_pose
        info["predicted_pose"] = self.predicted_pose
        info["pose_err_true_pred_translation"] = self.pose_err_true_pred.translation
        info["pose_err_true_pred_rotation"] = self.pose_err_true_pred.rotation
        info["pose_err_true_scan_match_translation"] = self.pose_err_true_scan_match.translation
        info["pose_err_true_scan_match_rotation"] = self.pose_err_true_scan_match.rotation

        return info
    

    def write_info_csv(self, info: dict, csv_file_path: str, headers: List[str]=None):
        '''
        Writes only the requested headers that are present in the info dictionary to a csv file.
        Parameters:
        -----------
        info: dict
            The info dictionary containing the data to be written to the csv file.
        headers: List[str]
            The list of headers for the CSV file.
        csv_file_path: str
            The path of the csv file where the data should be written to.
        '''
        # Add timestamp to info
        row_data = dict(info)
        current_time = rospy.get_time()
        row_data["timestamp"] = current_time

        # Filter headers if headers exist
        if headers is not None:
            fieldnames = [header for header in headers if header in row_data]

            if not fieldnames:
                rospy.logwarn(f"No matching CSV headers found in info.")
                return
            
            filtered_info = {header: row_data[header] for header in fieldnames}
        else:
            fieldnames = list(row_data.keys())
            filtered_info = row_data
        
        # Create parent dir if not existing
        parent_dir = os.path.dirname(csv_file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        file_exists = os.path.isfile(csv_file_path)

        # Write to file
        with open(csv_file_path, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
                rospy.loginfo(f"Created new csv file and wrote info to {csv_file_path}")
            else:
                rospy.loginfo(f"Appended info to existing csv file {csv_file_path}")

            writer.writerow(filtered_info)


    def link_state_cb(self, link_states: LinkStates):
        '''
        Receive gazebo link state from topic. Also find the ID corresponding to the link state which we wanne access.
        '''
        self.lock.acquire()
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
        
        self.lock.release()


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


    def publish_pose_err(self, topic_key: str, pose_err: Tuple[float, float]):
        '''
        Publishes the given pose error as a PoseErr2D message to the topic corresponding to the topic key.
        Parameters:
        -----------
        topic_key: str
            The key of the topic the messages should be published to.
        pose_err: Tuple[float, float]
            The pose error to be published, given as a tuple of (translation_error, rotation_error).
        '''
        # Extract pose error
        translation_err, rotation_err = pose_err

        # Define message
        msg = PoseErr2D()   
        msg.trans_err = translation_err
        msg.rot_err = rotation_err

        # Publish message
        self.publishers[topic_key].publish(msg)

    
    def transform_laser_scan_to_measurement(self, laser_scan: LaserScan) -> List[Tuple[float, float]]:
        '''
        Transforms the sensor msgs LaserScan to a list of measurement's consisting of (range, bearing) tuple. Only
        every nth measurement will be taken into account. 
        Also the range values will be filtered by the distances set in the algorithm.

        ->  A measurement with a value higher than the max range will be ignored (Sometimes range noise increase with
            high range values)
        -> A measurement with a value lower than the min range will be ignored (Sometimes range noise increase with low
            range values or the distance is in the robots chassis)

        Parameters:
        -----------
        laser_scan: LaserScan
            The laser scan message received from the topic. 
        
        Returns:
        --------
        List[Tuple[float, float]]: A list of measurements consisting of (range, bearing) tuples.
        '''
        min_angle= laser_scan.angle_min
        angle_increment= laser_scan.angle_increment
        min_range = max(laser_scan.range_min, self.scan_matcher.min_sensor_range)
        max_range = min(laser_scan.range_max, self.scan_matcher.max_sensor_range)
        bearing= min_angle
        measurements= []
        counter= 0
        skipped_invalid_measurements = 0
        # Transform LaserScan data
        for i in range(len(laser_scan.ranges)):
            # Only use every nth measurement
            if(not (counter % self.every_nth_ray)):
                r= laser_scan.ranges[i]
                if isfinite(r) and min_range <= r <= max_range:
                    measurements.append((r, bearing))
                else:
                    skipped_invalid_measurements += 1
            bearing+= angle_increment
            counter+= 1

        if skipped_invalid_measurements and not measurements:
            rospy.logwarn_throttle(5.0, "Laser scan contained no usable beams for scan matching.")

        return measurements
        
    
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
        position_error = sqrt((x_true - x_uncertain) ** 2 + (y_true - y_uncertain) ** 2)

        # Compute orientation error
        orientation_error = yaw_true - yaw_uncertain
        orientation_error = atan2(sin(orientation_error), cos(orientation_error))

        orientation_error_grad = orientation_error * 180 / pi

        return position_error, orientation_error, orientation_error_grad


    def execute(self):
        update_rate = rospy.Rate(self.update_rate)

        # TODO: Create init check for our scan matcher as a function

        while not rospy.is_shutdown():
            # Check if all necessary data is received
            if(
                self.link_state_message is not None and
                self.link_state_index is not None and
                self.laser_scan is not None and
                self.distance_left_wheel is not None and
                self.distance_right_wheel is not None
            ):
                # Check if scan matching is necessary
                min_dist = self.scan_matcher.ogm.grid_resolution_m
                if self.distance_left_wheel > min_dist or self.distance_right_wheel > min_dist:

                    # Lock threads
                    self.lock.acquire()

                    # Extract data
                    # Extract robot pose from link state message
                    link_state = self.link_state_message
                    link_state_index = self.link_state_index 
                    # Extract laser scan data
                    laser_scan = self.laser_scan
                    # Extract wheel encoder data
                    distance_left_wheel = self.distance_left_wheel
                    distance_right_wheel = self.distance_right_wheel
                    self.distance_left_wheel = 0.0
                    self.distance_right_wheel = 0.0

                    # Release lock
                    self.lock.release()

                    # Transform received data
                    # Transform link state to planar pose
                    self.true_pose = self.transform_link_state_pose_to_planar_pose(
                        link_state=link_state,
                        link_state_index=link_state_index
                    )
                    # Transform measurement to range bearing tuples
                    measurements = self.transform_laser_scan_to_measurement(laser_scan=laser_scan)

                    if len(measurements) < self.min_valid_measurements:
                        rospy.logwarn_throttle(
                            5.0,
                            f"Skipping scan matching because only {len(measurements)} valid beams are available.",
                        )
                        update_rate.sleep()
                        continue

                    # Correct pose by scan matching
                    self.scan_match_pose, self.predicted_pose = self.scan_matcher.update_pose(
                        old_pose=self.scan_match_pose,
                        dl=distance_left_wheel,
                        dr=distance_right_wheel,
                        measurements=measurements
                    )

                    # Compute pose error between true pose and predicted pose.
                    if self.predicted_pose is not None:
                        self.pose_err_true_pred.translation, rot_err_true_pred, self.pose_err_true_pred.rotation = self.compute_pose_err(
                            true_pose=self.true_pose,
                            unaccurate_pose=self.predicted_pose
                        )
                    
                        # Publish predicted pose
                        self.publish_pose(
                            topic_key=self.PRED_POSE,
                            pose=self.predicted_pose
                        )

                        # Publish pose error between true pose and predicted pose
                        self.publish_pose_err(
                            topic_key=self.POSE_ERR_TRUE_PRED,
                            pose_err=(self.pose_err_true_pred.translation, self.pose_err_true_pred.rotation)
                        )

                         # Log position and orientation error between true pose and predicted pose
                        rospy.loginfo("\n\nDisplay position and orientation error")
                        rospy.loginfo(f"True pose: {[f'{x:.2f}' for x in self.true_pose]}")
                        rospy.loginfo(f"Predicted pose: {[f'{x:.2f}' for x in self.predicted_pose]}")
                        rospy.loginfo(f"Pose error between true pose and predicted pose:")
                        rospy.loginfo(f"translation error = {self.pose_err_true_pred.translation:.4f}, Rotation error = {self.pose_err_true_pred.rotation:.4f} ")
                    
                    else:
                        rospy.loginfo("\nError occured in the pose update. Not predicted pose available.")

                    # Compute pose error between true pose from gaezebo link states and corrected pose by scan matching 
                    if self.scan_match_pose is not None:
                        self.pose_err_true_scan_match.translation, rot_error, self.pose_err_true_scan_match.rotation = self.compute_pose_err(
                            true_pose=self.true_pose,
                            unaccurate_pose=self.scan_match_pose
                        )

                        # Publish scan matching pose
                        self.publish_pose(
                            topic_key=self.SCAN_MATCH_POSE,
                            pose=self.scan_match_pose
                        )

                         # Publish pose error between true pose and scan matching pose
                        self.publish_pose_err(
                            topic_key=self.POSE_ERR_TRUE_SCAN_MATCH,
                            pose_err=(self.pose_err_true_scan_match.translation, self.pose_err_true_scan_match.rotation)
                        )

                        # Log position and orientation error between true pose and scan matching pose                                        
                        rospy.loginfo(f"\nTrue pose: {[f'{x:.2f}' for x in self.true_pose]}")
                        rospy.loginfo(f"Corrected pose: {[f'{x:.2f}' for x in self.scan_match_pose]}")
                        rospy.loginfo(f"Pose error between true pose and scan matching pose:")
                        rospy.loginfo(f"translation error = {self.pose_err_true_scan_match.translation:.4f}, Rotation error = {self.pose_err_true_scan_match.rotation:.4f} ")

                    else:
                        rospy.loginfo("\nError occured in the pose update. Not corrected pose available.")

                    
                    # Publish true pose
                    self.publish_pose(
                        topic_key=self.TRUE_POSE,
                        pose=self.true_pose
                    )

                    # Save info to csv
                    info = self.get_info()
                    self.write_info_csv(
                        info=info,
                        csv_file_path=self.storage_path,
                        headers=self.headers_to_write,
                    )
                               
            update_rate.sleep()


#__________________________________________________________________________________________________________________________________
#  Helper for map getting map from map server
#__________________________________________________________________________________________________________________________________

def get_occupancy_grid_map(map_service_name="static_map", service_class= GetMap):
    '''Calling the service and receives the map. Extracts the map data.'''
    occupancy_grid_map= None
    rospy.wait_for_service("static_map")
    try:
        map_loader= rospy.ServiceProxy("static_map", service_class)
        occupancy_grid_map= map_loader()
        return occupancy_grid_map.map
    except rospy.ServiceException() as e:
        rospy.loginfo("The Service %s failed", e)


def extract_map_meta_data(occupancy_grid_map):
    '''Returns the parameter of the occupancy grid map given the occupancy_grid_map Message object.'''
    frame_id= occupancy_grid_map.header.frame_id
    map_width= occupancy_grid_map.info.width
    map_height= occupancy_grid_map.info.height
    origin_x= occupancy_grid_map.info.origin.position.x
    origin_y= occupancy_grid_map.info.origin.position.y
    grid_resolution= occupancy_grid_map.info.resolution
    return (frame_id, map_width, map_height, origin_x, origin_y, grid_resolution)


def transform_2D_grid_to_1D_grid(self, indice):
    '''Transforms a given 2D grid cell indice to an 1D grid cell index.'''
    row, column= indice
    index= row * self.number_of_grids_x + column
    return int(index)


#__________________________________________________________________________________________________________________________________
#  Main
#__________________________________________________________________________________________________________________________________

def main():
    # Init node
    rospy.init_node("scan_matching_node", anonymous=True)

    # Headers to write
    headers_to_write = [
        "iteration",
        "mean_err",
        "rel_improvement",
        "no_improvement_counter",
        "min_mean_err",
        "dtrans_norm",
        "drot_abs",
        "stop_reason",
        "max_correspondence_distance",
        "min_squared_error",
        "n_points_true_data",
        "n_points_new_data",
        # "transformed_new_data_list",  
        # "squared_error_list",  
        # "transformation_parameter_list",  
        # "list_of_cleaned_corresp",  
        # "list_of_cleaned_corresp_numb", 
        # "scan_match_pose",
        # "true_pose",
        # "predicted_pose",
        "pose_err_true_pred_translation",
        "pose_err_true_pred_rotation",
        "pose_err_true_scan_match_translation",
        "pose_err_true_scan_match_rotation",
        "timestamp",
    ]


    # Define ros parameter
    # Define subscriber topics
    link_state_topic = "/gazebo/link_states"
    link_state_name = "robot_vacuum_cleaner::base_link"
    scan_topic= "scan"
    wheel_encoder_topic= "wheel_encoder"
    # Define update rate
    update_rate = 2.0
    # summarize ros parameter
    ros_parameter= (link_state_name, link_state_topic, scan_topic, wheel_encoder_topic, update_rate)

    every_nth_ray = 5   # Only use every nth ray of the laser scan for scan matching to reduce computational cost

    # Define robot params
    # Robot chassis parameter (need to be received from .yaml later)
    h_chassis= 0.15
    dist_chassis_to_ground= h_chassis/5
    r_wheel= h_chassis/2 + dist_chassis_to_ground
    w_wheel= 0.3 * r_wheel
    r_chassis= 0.25
    wheel_separation= 2 * r_chassis + w_wheel


    # Get map and extract infos
    # Get ogm from map server
    map_service_name="static_map"
    service_class= GetMap
    occupancy_grid_map_msg= get_occupancy_grid_map(map_service_name=map_service_name, service_class= service_class)
    # Extract map meta data from message
    frame_id, map_width, map_height, origin_x, origin_y, grid_resolution= extract_map_meta_data(occupancy_grid_map_msg)
    # Transform 1D map to 2D map
    occupancy_grid_map_2D= np.reshape(occupancy_grid_map_msg.data, (map_height, map_width))
        
    # Define start pose of robot 
    start_pose= (0.0, 0.0, 0.0)
    
    # Define parameter for ogm
    min_distance_to_border= 10.0    # The minimum distance from the actual robot pose to the border before extending the map

    # Define ogm param
    # Summarize map param
    map_parameters= (min_distance_to_border)
    # Define occupancy param
    prior_probability= 0.5          # Init map with probability of 0.5
    increasing_probability= 0.65
    decreasing_probability= 0.35
    max_log_odds= 100
    min_log_odds= -100
    occupancy_parameters= [prior_probability, increasing_probability, decreasing_probability, min_log_odds, max_log_odds]

    # Define ogm -> log Odds map transformation param
    occ = 100.0
    free = 0.0
    log_odds_occ = 100.0
    log_odds_free = -100.0
    log_odds_unknown = 0.0

    # Summarize params
    occ_params = (occ, free)
    log_odds_params = (log_odds_occ, log_odds_free, log_odds_unknown)

    # Define senor param 
    min_sensor_range= 0.1
    max_sensor_range= 10.0
    delta_r = 1.5
    sensor_parameters_ogm= (min_sensor_range, max_sensor_range)
    sensor_parameters_scan_matcher= (min_sensor_range, max_sensor_range, delta_r)
    # Define occ threshold for map point extraction for scan matching
    # Only cells > occ_thres will be considered as occupied and used for scan matching
    occ_thres = 50.0

    # Define icp scan matching param
    stop_params = {
        "max_iterations": 10,           
        "epsilon_rel": 1e-3,
        "no_improvement_limit": 3,
        "min_error": 5e-4,
        "epsilon_transform": 1e-4,
        "min_dtrans": 1e-3,
        "min_drot": 1e-2
    }
    # Max distance between two datapoints to build an correspondences set
    max_correspond_dist = 0.8

    # init icp 
    icp = IterativeClosestPoint(
        stop_params=stop_params,
        max_correspondence_distance=max_correspond_dist,
    )

    # Define robot parameter for scan matcher
    robot_parameter= (start_pose, wheel_separation)

    # Init ogm class
    ogm = OGM(
        map_parameter=map_parameters,
        occupancy_parameter=occupancy_parameters,
        sensor_parameter=sensor_parameters_ogm
    )

    # Transform ogm -> logOdds Map
    log_odds_map = ogm.transform_occupany_map_to_log_odds_map(
        ogm=occupancy_grid_map_2D,
        occ_params=occ_params,
        log_odds_param=log_odds_params, 
    )

    # Store map from map sever inside ogm class
    ogm.init_map_from_map(
        log_odds_map=log_odds_map,
        grid_resolution=grid_resolution
    )

    # Init scan matcher
    scan_matcher = ScanMatcher(
        ogm=ogm,
        icp=icp,
        robo_param=robot_parameter,
        sensor_parameters=  sensor_parameters_scan_matcher,
        occ_thres=occ_thres,
    )
    
    # Initialize node class
    scan_matching_node = ScanMatchingNode(
        ros_parameter=ros_parameter,
        scan_matcher=scan_matcher,
        every_nth_ray=every_nth_ray,
        headers_to_write=headers_to_write,
    )

    # Execute
    scan_matching_node.execute()


if __name__ == "__main__":
    main()