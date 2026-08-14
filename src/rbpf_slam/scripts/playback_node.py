#!/usr/bin/env python3

from typing import Tuple, Dict, Any, List

from dataclasses import dataclass
from math import atan2, cos, sin, sqrt
from pathlib import Path
import random
import threading
import time
import xml.etree.ElementTree as ET

import rospy
import message_filters
import tf2_ros

from geometry_msgs.msg import Pose
from geometry_msgs.msg import TransformStamped

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion, quaternion_from_euler

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
    from rbpf_slam.src.slam.infrastructure.defs import Pose2D

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
    from slam.infrastructure.defs import Pose2D


'''
Description
-----------
This node records synchronized laser scan and ground-truth pose data. It computes odometry controls
from the previous synchronized pose and adds artificial wheel encoder noise to them.

The synchronized data is stored in the configured playback directory. Three files are created:

    1. <timestamp>_steps.csv
        Contains the timestamp, left and right wheel travel, and true pose for each playback step.
    2. <timestamp>_scans.jsonl
        Contains the laser scan data for each playback step.
    3. <timestamp>_meta.json
        Contains playback metadata including the map name, robot start pose, sensor parameters, wheel
        separation, synchronization settings, and experiment tag.
'''

# Name of the ROS node
NODE_NAME = "playback_node"

# Root directory of the rbpf_slam package
RBPF_SLAM_DIR = Path(__file__).resolve().parents[1]


@dataclass
class ROSParams:
    '''
    Dataclass that defines the ROS and simulator parameters required by the playback node. These include
    topic names, synchronization settings, robot start pose, and noise metadata.
    '''
    ground_truth_topic: str
    scan_topic: str
    odom_tf_frame: str
    base_tf_frame: str
    desired_time_window_s: float
    time_window_tolerance_s: float
    max_sync_error_s: float
    time_synchronizer_queue_size: int
    time_synchronizer_slop_s: float
    robot_start_pose: Tuple[float, float, float]
    motion_error_factor: float
    turn_error_factor: float
    laser_range_resolution: float
    laser_noise_type: str
    laser_noise_mean: float
    laser_noise_stddv: float


@dataclass
class MetadataParams:
    '''
    Dataclass that defines the general metadata stored with each playback recording.
    '''
    map_name: str


@dataclass
class RECORDParams:
    '''
    Dataclass that defines the configuration values used by the PlaybackRecorder.
    '''
    enable_recording: bool
    output_dir: str


def load_wheel_separation() -> float:
    '''
    Loads the wheel separation computed in the generated robot description.

    Returns
    -------
    wheel_separation : float
        The distance between the left and right wheels in meters.

    Raises
    ------
    RuntimeError
        If the differential-drive plugin or its wheelSeparation element cannot be found in the robot
        description.
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



def load_experiment_params(
    robot_start_pose: Tuple[float, float, float],
    wheel_separation: float,
) -> ExperimentParams:
    '''
    Loads the playback experiment configuration from the ROS parameter server and adds robot-specific
    runtime values.

    Parameters
    ----------
    robot_start_pose : Tuple[float, float, float]
        The robot start pose in the map frame as an (x, y, theta) tuple.
    wheel_separation : float
        The wheel separation loaded from the generated robot description.

    Returns
    -------
    exp_params : ExperimentParams
        The experiment parameters stored as part of the playback metadata.

    Raises
    ------
    RuntimeError
        If the experiment configuration is missing required values or contains values that cannot
        initialize the parameter dataclasses.
    '''
    config = rospy.get_param("~experiment")

    try:
        return ExperimentParams(
            occupancy_params=OccupancyParams(
                **config["occupancy_params"]
            ),
            sensor_params=SensorParams(
                **config["sensor_params"]
            ),
            map_param=MapParameter(
                **config["map_params"]
            ),
            icp_params=ICPParams(
                **config["icp_params"]
            ),
            robot_params=RobotParams(
                wheel_separation=wheel_separation,
            ),
            scan_matcher_params=ScanMatcherParams(
                **config["scan_matcher_params"]
            ),
            particle_params=ParticleParams(
                start_pose=robot_start_pose,
                **config["particle_params"],
            ),
            motion_model_params=MotionModelParams(
                wheel_separation=wheel_separation,
                **config["motion_model_params"],
            ),
            measurement_model_params=MeasurementModelParams(
                **config["measurement_model_params"]
            ),
            tag=str(config["tag"]),
            **config["rbpf_params"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid playback experiment configuration: {exc}"
        ) from exc



def load_metadata_params() -> MetadataParams:
    '''
    Loads the general playback metadata settings from the ROS parameter server and initializes the
    MetadataParams dataclass.

    Returns
    -------
    metadata_params : MetadataParams
        The loaded metadata settings for the playback recording.

    Raises
    ------
    RuntimeError
        If the metadata configuration is missing required values or has an invalid structure.
    '''
    config = rospy.get_param("~metadata")

    try:
        return MetadataParams(
            map_name=str(config["map_name"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid playback metadata configuration: {exc}"
        ) from exc


def load_record_params() -> RECORDParams:
    '''
    Loads the recorder settings from the ROS parameter server and initializes the RECORDParams dataclass.

    Returns
    -------
    record_params : RECORDParams
        The loaded configuration values for the PlaybackRecorder.

    Raises
    ------
    RuntimeError
        If the recorder configuration is missing required values or has an invalid structure.
    '''
    config = rospy.get_param("~recording")

    try:
        output_dir = Path(str(config["output_dir"])).expanduser()
        if not output_dir.is_absolute():
            output_dir = RBPF_SLAM_DIR / output_dir

        return RECORDParams(
            enable_recording=bool(config["enable_recording"]),
            output_dir=str(output_dir.resolve()),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid playback recording configuration: {exc}"
        ) from exc


def load_ros_node_params(
    robot_start_pose: Tuple[float, float, float],
    motion_error_factor: float,
    turn_error_factor: float,
    laser_range_resolution: float,
    laser_noise_type: str,
    laser_noise_mean: float,
    laser_noise_stddv: float,
) -> ROSParams:
    '''
    Loads fixed ROS settings from the ROS parameter server and adds the dynamic values supplied by the
    launch file and simulator.

    Parameters
    ----------
    robot_start_pose : Tuple[float, float, float]
        The robot start pose supplied by the launch file.
    motion_error_factor : float
        The motion-dependent wheel encoder noise factor.
    turn_error_factor : float
        The turn-dependent wheel encoder noise factor.
    laser_range_resolution : float
        The configured laser range resolution in meters.
    laser_noise_type : str
        The type of noise applied by the simulated laser scanner.
    laser_noise_mean : float
        The mean of the simulated laser noise.
    laser_noise_stddv : float
        The standard deviation of the simulated laser noise.

    Returns
    -------
    ros_params : ROSParams
        The ROS parameters initialized with fixed configuration and dynamic runtime values.

    Raises
    ------
    RuntimeError
        If the ROS configuration is missing required values or contains values that cannot be converted
        to the expected types.
    '''
    config = rospy.get_param("~ros")

    try:
        topics = config["topics"]
        frames = config["frames"]
        synchronization = config["synchronization"]

        return ROSParams(
            ground_truth_topic=topics["ground_truth"],
            scan_topic=topics["scan"],
            odom_tf_frame=frames["odom"],
            base_tf_frame=frames["base"],
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
            laser_range_resolution=laser_range_resolution,
            laser_noise_type=laser_noise_type,
            laser_noise_mean=laser_noise_mean,
            laser_noise_stddv=laser_noise_stddv,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid playback ROS configuration: {exc}"
        ) from exc


def build_metadata(
    metadata_params: MetadataParams,
    exp_params: ExperimentParams,
    ros_params: ROSParams,
) -> Dict:
    '''
    Builds the metadata dictionary stored with the recorded playback data.

    Parameters
    ----------
    metadata_params : MetadataParams
        General metadata settings for the recording.
    exp_params : ExperimentParams
        Experiment, robot, map, and sensor parameters associated with the recording.
    ros_params : ROSParams
        ROS, synchronization, start-pose, and simulator-noise parameters.

    Returns
    -------
    metadata : Dict
        The complete metadata dictionary written by the PlaybackRecorder.
    '''
    return {
        "map": metadata_params.map_name,

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
        "time_synchronizer_slop": ros_params.time_synchronizer_slop_s,

        "ground_truth_topic": ros_params.ground_truth_topic,
        "scan_topic": ros_params.scan_topic,

        "data_time_seperation_s": ros_params.desired_time_window_s,
        "d_time_window": ros_params.time_window_tolerance_s,
        "max_sync_error_s": ros_params.max_sync_error_s,

        "n_particles": exp_params.particle_params.n_particles,
        "comment": exp_params.tag,
    }



class ROSPlaybackNode:
    '''
    ROS interface for recording synchronized playback data for offline RBPF SLAM evaluation.

    Integrates the PlaybackRecorder into the robot's ROS infrastructure. The node receives synchronized
    laser scans and ground-truth odometry, derives noisy wheel encoder controls, and stores each accepted
    pair as one playback step together with the configured metadata.

    Processing pipeline
    -------------------
    1. Receive the synchronized laser scan and odometry data.
    2. Validate the recording interval and synchronization error.
    3. Compute left and right wheel travel from consecutive poses.
    4. Add artificial wheel encoder noise.
    5. Store the scan, controls, and synchronized true pose.

    Parameters
    ----------
    metadata_params : MetadataParams
        General metadata settings stored with the playback recording.
    ros_params : ROSParams
        ROS topics, synchronization settings, robot pose, and simulator noise.
    record_params : RECORDParams
        Configuration values used to initialize the PlaybackRecorder.
    exp_param : ExperimentParams
        Experiment parameters associated with the recorded data.
    '''

    def __init__(
        self,
        metadata_params: MetadataParams,
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
            metadata_params=metadata_params,
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

        # Define TF infra
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Define shutdown behavior
        rospy.on_shutdown(self.on_shutdown)


    def on_shutdown(self) -> None:
        '''Logs the shutdown of the synchronized playback node.'''
        rospy.loginfo("Shutting down synchronized playback node.")


    @staticmethod
    def _pose_into_transform_stamped_msg(
        pose: Pose2D,
        parent_frame: str,
        child_frame: str,
        timestamp: rospy.Time,
    ) -> TransformStamped:
        """
        Converts a 2D pose into a ROS TransformStamped message. The pose
        describes the child frame relative to the parent frame.

        Parameters
        ----------
        pose : Pose2D
            A tuple representing the 2D pose (x, y, theta), where theta is the
            orientation in radians.
        parent_frame : str
            The frame ID of the parent frame.
        child_frame : str
            The frame ID of the child frame.
        timestamp : rospy.Time
            The timestamp for the TransformStamped message.

        Returns
        -------
        TransformStamped
            A ROS TransformStamped message containing the given pose, frame
            IDs, and timestamp.
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

    
    def synchronizer_cb(
        self, 
        laser_scan: LaserScan,
        ground_truth_odom: Odometry,
    ) -> None:
        '''
        Processes and records a synchronized laser scan and ground-truth odometry pair.

        The callback checks the elapsed time since the previously accepted scan and the synchronization
        error between both messages. Valid pairs are converted into noisy wheel controls and stored as
        one playback step; invalid pairs are skipped.

        Parameters
        ----------
        laser_scan : LaserScan
            The synchronized laser scan message.
        ground_truth_odom : Odometry
            The synchronized ground truth odometry message.
        '''
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
        if time_diff > (
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
            # rospy.loginfo(
            #     f"True pose: x={pose[0]:.2f}, y={pose[1]:.2f}, yaw={pose[2]:.2f}"
            # )
            
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

            # Compute TFs
            # Define tf odom -> base link
            odom_base_tf = self._pose_into_transform_stamped_msg(
                pose=pose,
                parent_frame=self.ros_params.odom_tf_frame,
                child_frame=self.ros_params.base_tf_frame,
                timestamp=laser_scan_cp.header.stamp,
            )

            # Publish tf odom -> base link
            self.tf_broadcaster.sendTransform(odom_base_tf)                

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
    def wheelencoder_simulation(old_pose, new_pose, width, eps_alpha= 1e-3) -> Tuple[float, float]:
        '''
        Computes the left and right wheel travel between two planar robot poses using differential-drive
        kinematics.

        Parameters
        ----------
        old_pose : Tuple[float, float, float]
            The previous robot pose as an (x, y, theta) tuple.
        new_pose : Tuple[float, float, float]
            The current robot pose as an (x, y, theta) tuple.
        width : float
            The distance between the left and right wheels in meters.
        eps_alpha : float, optional
            The minimum heading change treated as a turn; defaults to 1e-3 radians.

        Returns
        -------
        left_control : float
            The distance traveled by the left wheel in meters.
        right_control : float
            The distance traveled by the right wheel in meters.
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

        The variance follows the previous wheel encoder model: motion noise depends on each wheel
        distance, while turn noise depends on their difference.

        Parameters
        ----------
        left_control : float
            The ideal distance traveled by the left wheel in meters.
        right_control : float
            The ideal distance traveled by the right wheel in meters.

        Returns
        -------
        noisy_left_control : float
            The left wheel travel after applying artificial noise.
        noisy_right_control : float
            The right wheel travel after applying artificial noise.
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
        Converts a ROS Pose message into a planar pose tuple.

        Parameters
        ----------
        pose : Pose
            The ROS pose message to convert.

        Returns
        -------
        planar_pose : Tuple[float, float, float]
            The pose represented as an (x, y, yaw) tuple.
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
        '''
        Keeps the node alive while the registered callback processes synchronized input.
        '''
        rospy.spin()



def main():
    '''
    Initializes the synchronized playback node. The following steps are performed:
        1. Initialize the ROS node.
        2. Load dynamic robot and simulator values.
        3. Load the playback configuration from the ROS parameter server.
        4. Initialize the playback recorder and ROS interface.
        5. Start processing synchronized input data.
    '''
    # Init node
    rospy.init_node(NODE_NAME)
    config_name = rospy.get_param("~config_name")

    # Get motion error parameters
    motion_error_factor = float(rospy.get_param(
        "/motion_error_factor"
    ))
    turn_error_factor = float(rospy.get_param(
        "/turn_error_factor"
    ))

    # Get robot spawn pose
    robot_start_pose = (
        float(rospy.get_param("/spawn_x")),
        float(rospy.get_param("/spawn_y")),
        float(rospy.get_param("/spawn_yaw")),
    )

    # Get laser noise parameters
    laser_range_resolution = float(
        rospy.get_param("/laser_range_resolution")
    )
    laser_noise_type = str(rospy.get_param("/laser_noise_type"))
    laser_noise_mean = float(rospy.get_param("/laser_noise_mean"))
    laser_noise_stddv = float(rospy.get_param("/laser_noise_stddv"))

    # Load YAML configuration and add runtime robot/simulator parameters
    wheel_separation = load_wheel_separation()
    exp_param = load_experiment_params(
        robot_start_pose=robot_start_pose,
        wheel_separation=wheel_separation,
    )
    metadata_params = load_metadata_params()
    rec_params = load_record_params()
    ros_params = load_ros_node_params(
        robot_start_pose=robot_start_pose,
        motion_error_factor=motion_error_factor,
        turn_error_factor=turn_error_factor,
        laser_range_resolution=laser_range_resolution,
        laser_noise_type=laser_noise_type,
        laser_noise_mean=laser_noise_mean,
        laser_noise_stddv=laser_noise_stddv,
    )

    # Display loaded parameters 
    rospy.loginfo("Loaded playback configuration: %s", config_name)
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
        metadata_params=metadata_params,
        ros_params=ros_params,
        record_params=rec_params,
        exp_param=exp_param,
    )

    # Run node
    playback_node.exe()


if __name__ == "__main__":
    main()
