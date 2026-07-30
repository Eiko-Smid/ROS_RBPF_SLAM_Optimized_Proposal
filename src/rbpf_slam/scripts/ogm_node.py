#!/usr/bin/env python3
import debugpy

import rospy
import tf2_ros

from queue import Empty, Full, Queue

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose2D as Pose2DMsg, TransformStamped
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from rbpf_slam.msg import LogOddsMap, RBPFInput


from typing import Tuple
from dataclasses import dataclass
import numpy as np

# Import classes (support both roslaunch and direct execution contexts)
try:
    from rbpf_slam.src.slam.scan_matcher.ogm_scan_matching import OGM
    from rbpf_slam.src.slam.rbpf.scan_match_factory import (
        OccupancyParams,
        SensorParams,
        MapParameter,
    )
except ModuleNotFoundError:
    from slam.scan_matcher.ogm_scan_matching import OGM
    from slam.rbpf.scan_match_factory import (
        OccupancyParams,
        SensorParams,
        MapParameter,
    )

USE_DEBUGGER = False
Pose2D = Tuple[float, float, float]


def debug_code():
    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()
    print("Debugger attached")



@dataclass
class ROSParams:
    rbpf_input_topic = "rbpf/input"
    input_queue_size = 10
    odom_tf_frame = "odom_link"
    base_tf_frame = "base_link"
    laser_tf_frame = "laser_scanner_link"
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
            increasing_probability=0.85,
            decreasing_probability=0.15,
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

        self.laser_pose_world = None

        # Cache the static base->laser transform once (2D: x, y, yaw).
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.base_to_laser_pose_2d = self.lookup_base_to_laser_transform_2d()
        
        # Receive synchronized scan, wheel odometry, and true pose data from the
        # RBPF data processor.
        self.rbpf_input_queue = Queue(maxsize=self.ros_params.input_queue_size)
        self.rbpf_input_subscriber = rospy.Subscriber(
            name=self.ros_params.rbpf_input_topic,
            data_class=RBPFInput,
            callback=self._rbpf_input_cb,
            queue_size=5,
        )

        # Define publisher for map
        self.map_publisher = rospy.Publisher(self.ros_params.map_topic, LogOddsMap, queue_size=1) 

        # Colorization
        self.col_map_poitns = rospy.Publisher("debug_cells", Marker, queue_size=1)


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


    def _rbpf_input_cb(self, msg: RBPFInput) -> None:
        '''Store synchronized input data for processing in the main loop.'''
        try:
            self.rbpf_input_queue.put_nowait(msg)
        except Full:
            rospy.logerr_throttle(
                period=2.0,
                msg=(
                    "\nOGM input queue is full. The mapper cannot process "
                    "the incoming data fast enough!"
                ),
            )


    @staticmethod
    def _pose_into_transform_stamped_msg(
        pose: Pose2D,
        parent_frame: str,
        child_frame: str,
        timestamp: rospy.Time,
    ) -> TransformStamped:
        """
        Convert a 2D pose into a ROS TransformStamped message.

        The pose describes the child frame relative to the parent frame.
        """
        if pose is None or len(pose) != 3:
            raise ValueError("Pose must contain (x, y, theta).")

        transform_msg = TransformStamped()

        # Header
        transform_msg.header.stamp = timestamp
        transform_msg.header.frame_id = parent_frame

        # Frame located relative to the parent
        transform_msg.child_frame_id = child_frame

        # Translation of child inside parent
        transform_msg.transform.translation.x = float(pose[0])
        transform_msg.transform.translation.y = float(pose[1])
        transform_msg.transform.translation.z = 0.0

        # Rotation of child inside parent
        quat = quaternion_from_euler(
            0.0,
            0.0,
            float(pose[2]),
        )

        transform_msg.transform.rotation.x = quat[0]
        transform_msg.transform.rotation.y = quat[1]
        transform_msg.transform.rotation.z = quat[2]
        transform_msg.transform.rotation.w = quat[3]

        return transform_msg

    
    @staticmethod
    def transform_pose_to_planar_pose(pose: Pose2DMsg):
        '''
        Transform a ROS Pose2D message to an (x, y, yaw) tuple.
        '''
        return (pose.x, pose.y, pose.theta)


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
        self.map_publisher.publish(self.ogm.get_log_odds_map_object())



    def publish_green_cells(self, i_range, j_range, ogm):        
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()

        marker.ns = "debug"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD

        # IMPORTANT
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0

        res = ogm.grid_resolution_m
        marker.scale.x = res
        marker.scale.y = res
        marker.scale.z = 0.01

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # ----------- CORE LOGIC -----------
        i_min, i_max = i_range   # rows → y
        j_min, j_max = j_range   # cols → x

        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                x, y = ogm.transform_grid_cell_to_point((i, j))

                p = Point()
                p.x = x
                p.y = y
                p.z = 0.0

                marker.points.append(p)
        # ----------------------------------

        self.col_map_poitns.publish(marker)

        
    def execute(self):
        '''Main loop for executing the algorithm.'''
        while not rospy.is_shutdown():
            try:
                msg: RBPFInput = self.rbpf_input_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                rospy.loginfo_once("OGM Initalized.")

                # Extract the synchronized data produced by rbpf_data_processor_node.
                laser_scan = msg.laser_scan
                pose = self.transform_pose_to_planar_pose(msg.true_pose)

                # Optional world-frame laser pose for later world-point projection.
                self.laser_pose_world = self.transform_planar_pose(
                    pose,
                    self.base_to_laser_pose_2d,
                )

                measurements = self.transform_laser_scan_to_measurement(laser_scan)

                # Subsample measurements for faster processing (optional)
                subsample_factor = 5
                measurements = measurements[::subsample_factor]

                # Increase map size if necessary
                extension_needed= True
                while(extension_needed):
                    extension_needed= self.ogm.map_extension_if_necessary(self.laser_pose_world)

                # Update the map
                self.ogm.update_map(measurements, self.laser_pose_world)

                # log beam otuside map count
                if self.ogm.beam_out_map_count > 0:
                    rospy.loginfo(f"Beam outside map count: {self.ogm.beam_out_map_count}")

                # Define tf odom -> base link
                odom_base_tf = self._pose_into_transform_stamped_msg(
                    pose=pose,
                    parent_frame=self.ros_params.odom_tf_frame,
                    child_frame=self.ros_params.base_tf_frame,
                    timestamp=msg.header.stamp,
                )

                # Publish tfs
                self.tf_broadcaster.sendTransform(odom_base_tf)
                
                # Transform and publish map
                self.publish_occupancy_grid_message()

            finally:
                self.rbpf_input_queue.task_done()



def main():
    # Debug code if enabled
    if USE_DEBUGGER:
            debug_code()
            
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
