


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
    from rbpf_slam.src.slam.infrastructure.playback_recorder import PlaybackRecorder
    
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
    from slam.infrastructure.playback_recorder import PlaybackRecorder

    
from rbpf_slam.msg import WheelEncoder
from rbpf_slam.msg import PoseErr2D

NODE_NAME = "playback_node"
PLAYBACK_DIR = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/python_playback/"


def build_metadata(exp_params):
    return {
        # "map_resolution": exp_params.map_param.grid_resolution_m,
        # "map_width": exp_params.map_param.map_width,
        # "map_height": exp_params.map_param.map_height,
        "sensor_range_max": exp_params.sensor_params.max_sensor_range,
        "sensor_range_min": exp_params.sensor_params.min_sensor_range,
        "wheel_separation": exp_params.robot_params.wheel_separation,
        "wheel_encoder_sim_motion_error_factor": 0.1,
        "wheel_encoder_sim_turn_error_factor": 0.15,
        "n_particles": exp_params.particle_params.n_particles,
        "comment": exp_params.tag,
    }


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
                tag=("First rbpf node run"),
            )

    return exp_param


@dataclass
class RECORDParams:
    enable_recording: bool = True
    output_dir: str = PLAYBACK_DIR


class ROSPlaybackNode:
    def __init__(self, ros_params: ROSParams, record_params: RECORDParams, exp_param: ExperimentParams):
        # Store members
        self.record_params = record_params  
        self.ros_params = ros_params  
        self.exp_params = exp_param
        self.time_jumps = 0.0    

        # Build metadata
        metadata = build_metadata(exp_param)

        # Init recorder 
        self.recorder = PlaybackRecorder(
            output_dir=record_params.output_dir,
            metadata=metadata,
        )

        # Odometry: Distance of left and right wheel
        self.dl = 0.0
        self.dr = 0.0

        # Define link states
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

        # Define thread locker obj
        self.lock = threading.Lock()

        # Define shutdown behavior
        rospy.on_shutdown(self.on_shutdown)



    def on_shutdown(self):
        rospy.loginfo("Shutting down RBPF ROS node.")


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
    

    def exe(self):
        update_rate = rospy.Rate(self.ros_params.update_rate)
        while not rospy.is_shutdown():
            try:
                # Check if all necessary data is received
                min_dist = self.exp_params.map_param.grid_resolution_m
                if(
                    self.link_state_message is not None and
                    self.link_state_idx is not None and
                    self.laser_scan is not None and
                    (self.dl > min_dist or
                    self.dr > min_dist)
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
                    
                    # Transform 3D pose -> 2D pose
                    true_pose = self.transform_link_state_pose_to_planar_pose(
                        link_state=link_state,
                        link_state_index=link_idx
                    )

                    # Record step
                    self.recorder.record_step(
                        t=time.perf_counter(),
                        t_ros = rospy.get_time(),
                        dl=dl,
                        dr=dr,
                        true_pose=true_pose,
                        laser_scan=laser_scan
                    )
                

            except rospy.exceptions.ROSTimeMovedBackwardsException:
                rospy.logwarn("Time jump detected → skipping this iteration")
                self.time_jumps += 1
                continue

            finally:
                update_rate.sleep()


def main():
    # Parameters
    exp_param = define_exp_parameter()
    ros_params = ROSParams()
    rec_params = RECORDParams()

    # Init class
    playback_node = ROSPlaybackNode(
        ros_params=ros_params,
        record_params=rec_params,
        exp_param=exp_param,
    )

    # Init Node
    rospy.init_node(NODE_NAME)

    # Run node
    playback_node.exe()



if __name__ == "__main__":
    main()