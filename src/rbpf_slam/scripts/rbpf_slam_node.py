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
from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from rbpf_slam.msg import Measurement
from rbpf_slam.msg import LogOddsMap

from dataclasses import dataclass
import time
import numpy as np

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
TODO

1. Create first running version of ros communication (Status: )

2. Implement filter steps in execute (Status: )

3. Implement TFs (map -> odom) (Status: )

4. Implement ros topic for pose errors and display as rqt plot (Status: )

5. Implement playback (Status: )



'''


NODE_NAME = "rbpf_slam_node"

# Pose names
TRUE_POSE_TOPIC = "true_pose"
BEST_P_POSE = "best_particle_pose"
WEIGHTED_MEAN_P_POSE = "weighted_mean_particle_pose"
POSE_ERR_TRUE_BEST_P_TOPIC = "pose_err_true_best_p"
POSE_ERR_TRUE_MEAN_P = "pose_err_true_maen_p"


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
    map_tf_frame = "map"
    odom_tf_frame = "odom_link"
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
                every_nth_scan_filter=4,
                every_nth_scan_map=2,
                proposal_sigma_xy=0.1,
                proposal_sigma_theta=0.05,
                proposal_n_samples=10,
                tag=("First rbpf node run"),
            )

    return exp_param


class RBPFROS:
    def __init__(self, rbpf: RBPF, ros_params: ROSParams, exp_param: ExperimentParams):
        # Init members
        self.rbpf: RBPF = rbpf
        self.ros_params = ros_params
        self.exp_params = exp_param

        # Distance of left and right wheel
        self.dl = 0.0
        self.dr = 0.0

        # Defien link states
        self.link_state_message = None
        self.link_state_name = ros_params.link_state_name
        self.link_state_idx = None
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

        # Define pose publisher
        self.publishers = {
            WEIGHTED_MEAN_P_POSE: rospy.Publisher(
                name=WEIGHTED_MEAN_P_POSE,
                data_class=Pose,
                queue_size=2
            ),
            BEST_P_POSE: rospy.Publisher(
                name=BEST_P_POSE,
                data_class=Pose,
                queue_size=2
            ),
            TRUE_POSE_TOPIC: rospy.Publisher(
                name=TRUE_POSE_TOPIC,
                data_class=Pose,
                queue_size=2
            ),
            POSE_ERR_TRUE_BEST_P_TOPIC: rospy.Publisher(
                name=POSE_ERR_TRUE_BEST_P_TOPIC,
                data_class=PoseErr2D,
                queue_size=2,
            ),
            POSE_ERR_TRUE_MEAN_P: rospy.Publisher(
                name=POSE_ERR_TRUE_MEAN_P,
                data_class=PoseErr2D,
                queue_size=2,
            )
        }

        # Defien thread lcoker obj
        self.lock = threading.Lock()


        # Cache the static base->laser transform (x, , theta)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
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
            if self.link_state_idx is None:
                try:
                    self.link_state_idx = link_states.name.index(self.link_state_name)
                    rospy.loginfo(f"Found link state index: {self.link_state_idx}")
                except ValueError:
                    rospy.logwarn_throttle(5.0, f"Link {self.link_state_name} not found in Gazebo link states.")

            if self.link_state_idx is None:
                for i in range(len(link_states.name)):
                    if self.link_state_name == link_states.name[i]:
                        self.link_state_idx = i
                        break
    

    def laser_scan_cb(self, laser_scan):
        '''Receive laser scan from topic.'''
        self.lock.acquire()
        self.laser_scan= laser_scan
        self.lock.release()


    def wheel_encoder_cb(self, distance):
        '''Accumulate the distances of the left and right wheel.'''
        self.lock.acquire()
        self.dl+= distance.left
        self.dr+= distance.right
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
        min_angle = laser_scan.angle_min
        angle_increment = laser_scan.angle_increment
        bearing = min_angle
        measurements = []
        
        # Transform LaserScan data
        for range in laser_scan.ranges:
            measurement = (range, bearing)
            measurements.append(measurement)
            bearing += angle_increment
            
        return measurements   


    def publish_map(self, log_odds_map: LogOddsMap):
        '''Get's the logOdds map and do all necessary transformation's for publishing the Occupancy Message.'''
        # Publish current map from ogm algorithm        
        self.map_publisher.publish(log_odds_map)

    
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
        orientation = Quaternion(x=roll, y=pitch, z=yaw, w=w)
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


    @staticmethod
    def compute_pose_err(true_pose, unaccurate_pose):
        '''
        Computes the error between the true pose and the uncertain pose, reported by the scan matcher.

        Returns
        --------
        position_error: float
            The position error in meters.
        orientation_error: float
            The orientation error in radians.
        orientation_error_grad: float
            The orientation error in degree.
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


    @staticmethod
    def normalize_angle(angle: float) -> float:
        return np.arctan2(np.sin(angle), np.cos(angle))


    def publish_map_to_odom_tf(self, map_to_base_pose: Tuple[float, float, float]):
        """
        Publishes map -> odom using the current odom -> base_link TF and a map -> base_link pose estimate.
        """
        try:
            # get tf: odom -> base_link
            transform_odom_to_base = self.tf_buffer.lookup_transform(
                self.ros_params.odom_tf_frame,
                self.ros_params.base_tf_frame,
                rospy.Time(0),
                rospy.Duration(0.1),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            rospy.logwarn_throttle(
                2.0,
                "RBPF: could not lookup TF %s -> %s",
                self.ros_params.odom_tf_frame,
                self.ros_params.base_tf_frame,
            )
            return

        odom_x = transform_odom_to_base.transform.translation.x
        odom_y = transform_odom_to_base.transform.translation.y
        rotation = transform_odom_to_base.transform.rotation
        (_, _, odom_theta) = euler_from_quaternion(
            [rotation.x, rotation.y, rotation.z, rotation.w]
        )

        # RBPF estimate: map -> base_link (weighted mean)
        pose_x, pose_y, pose_theta = map_to_base_pose

        # Compute map -> odom from map -> base_link and odom -> base_link
        map_to_odom_theta = self.normalize_angle(pose_theta - odom_theta)
        c = np.cos(map_to_odom_theta)
        s = np.sin(map_to_odom_theta)

        map_to_odom_x = pose_x - (c * odom_x - s * odom_y)
        map_to_odom_y = pose_y - (s * odom_x + c * odom_y)

        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, map_to_odom_theta)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.header.frame_id = self.ros_params.map_tf_frame
        tf_msg.child_frame_id = self.ros_params.odom_tf_frame
        tf_msg.transform.translation.x = map_to_odom_x
        tf_msg.transform.translation.y = map_to_odom_y
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tf_msg)


    def exe(self):
        update_rate = rospy.Rate(self.ros_params.update_rate)

        rospy.loginfo(
            "Waiting for TF %s -> %s...",
            self.ros_params.odom_tf_frame,
            self.ros_params.base_tf_frame,
        )
        try:
            self.tf_buffer.lookup_transform(
                self.ros_params.odom_tf_frame,
                self.ros_params.base_tf_frame,
                rospy.Time(0),
                rospy.Duration(5.0),
            )
            rospy.loginfo("TF available!")
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            rospy.logwarn("Initial TF lookup timeout. Continuing and retrying during runtime.")

        while not rospy.is_shutdown():
            t_iter_start = time.perf_counter()
            try:
                # Check if all necessary data is received
                if(
                    self.link_state_message is not None and
                    self.link_state_idx is not None and
                    self.laser_scan is not None and
                    self.dl > 0.0 and
                    self.dr > 0.0 
                ):

                    # Extract data and reset data
                    with self.lock:
                        link_state = self.link_state_message
                        link_idx = self.link_state_idx
                        laser_scan = self.laser_scan
                        dl = self.dl
                        dr = self.dr

                        self.link_state_message = None
                        self.link_state_idx = None
                        self.laser_scan = None
                        self.dl = 0.0
                        self.dr = 0.0

                    # Get true pose
                    true_pose = self.transform_link_state_pose_to_planar_pose(link_state, link_idx)
                    
                    # Get measurement
                    measurement = self.transform_laser_scan_to_measurement(laser_scan)

                    # Define measurements for different algorithm parts
                    measurement_filter = measurement[::self.exp_params.every_nth_scan_filter]
                    measurement_map = measurement[::self.exp_params.every_nth_scan_map]

                    # Run RBPF step
                    self.rbpf.step(
                        odom=(dl, dr),
                        measurements_proposal=measurement_filter,
                        measurements_map_update=measurement_map,
                        true_pose=true_pose,
                        proposal_sigma_xy=self.exp_params.proposal_sigma_xy,
                        proposal_sigma_theta=self.exp_params.proposal_sigma_theta,
                        proposal_n_samples=self.exp_params.proposal_n_samples,
                    )

                    # Extract step info
                    info = self.rbpf.get_step_info()
                    best_p_idx: int = info.get("best_particle_idx")
                    best_particle_pose = info.get("best_particle_pose")
                    weighted_mean_pose = info.get("weighted_mean_pose")

                    # Compute pose errors
                    pose_err_true_best_p, _, Orient_err_true_best_p = self.compute_pose_err(
                        true_pose=true_pose,
                        unaccurate_pose=best_particle_pose,
                    )

                    pose_err_true_mean_p, _, Orient_err_true_mean_p = self.compute_pose_err(
                        true_pose=true_pose,
                        unaccurate_pose=weighted_mean_pose,
                    )

                    # Extract map
                    occ_map = self.rbpf.particles[best_p_idx].scan_matcher.get_ogm()                                        
                    
                    # Publish data
                    self.publish_map_to_odom_tf(map_to_base_pose=weighted_mean_pose)
                    self.publish_map(occ_map)
                    self.publish_pose(topic_key=TRUE_POSE_TOPIC, pose=true_pose)
                    self.publish_pose(topic_key=BEST_P_POSE, pose=best_particle_pose)
                    self.publish_pose(topic_key=WEIGHTED_MEAN_P_POSE, pose=weighted_mean_pose)
                    self.publish_pose_err(
                        topic_key=POSE_ERR_TRUE_BEST_P_TOPIC,
                        pose_err=(pose_err_true_best_p, Orient_err_true_best_p)
                    )
                    self.publish_pose_err(
                        topic_key=POSE_ERR_TRUE_MEAN_P,
                        pose_err=(pose_err_true_mean_p, Orient_err_true_mean_p)
                    )

            except rospy.exceptions.ROSTimeMovedBackwardsException:
                rospy.logwarn("Time jump detected → skipping this iteration")
                self.time_jumps += 1
                continue
            finally:
                iteration_time_ms = (time.perf_counter() - t_iter_start) * 1000.0
                rospy.loginfo("RBPF full iteration time: %.3f ms", iteration_time_ms)
                update_rate.sleep()



def main():
    # Parameters
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
    
    rbpf_ros = RBPFROS(
        rbpf=rbpf,
        ros_params=ros_params,
        exp_param=exp_param
    )

    # Run algorithm
    rbpf_ros.exe()

    

if __name__=="__main__":
    main()  