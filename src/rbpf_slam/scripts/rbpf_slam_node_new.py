#!/usr/bin/env python3
import debugpy
import traceback
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional, Dict, Any, List, Union

import geometry_msgs
import rospy
import threading
import tf2_ros

from queue import Queue, Empty, Full
from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# Import messages
from rbpf_slam.msg import RBPFInput
# from rbpf_slam.msg import LogOddsMap
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose2D as Pose2DMsg


from dataclasses import dataclass, field
import time
import numpy as np

# Import code infra

try: 
    from rbpf_slam.src.slam.rbpf.rbpf import (
        RBPFFactory,
        ParticleParams,
        MotionModelParams,
        BeamRangeFinderMeasModelParams,
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
    from rbpf_slam.src.slam.scan_matcher.ogm_scan_matching import OGM
    from rbpf_slam.src.slam.rbpf.motion_model import MotionModel
    from rbpf_slam.src.slam.optimize_rbpf_multiple_particles.playback_defs import ExperimentParams, PlaybackData    
    from rbpf_slam.src.slam.rbpf.raw_odom_estimator import RawOdomEstimator
    from rbpf_slam.src.slam.rbpf.rbpf_node_evaluator import RBPFEvaluator, StepResult
    from rbpf_slam.src.slam.infrastructure.transformations_2d import Transformations2D
    from rbpf_slam.src.slam.infrastructure.defs import Pose2D
    
except ModuleNotFoundError:
    from slam.rbpf.rbpf import (
        RBPFFactory,
        ParticleParams,
        MotionModelParams,
        BeamRangeFinderMeasModelParams,
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
    from slam.scan_matcher.ogm_scan_matching import OGM
    from slam.rbpf.motion_model import MotionModel
    from slam.optimize_rbpf_multiple_particles.playback_defs import ExperimentParams, PlaybackData
    from slam.rbpf.raw_odom_estimator import RawOdomEstimator
    from slam.rbpf.rbpf_node_evaluator import RBPFEvaluator, StepResult
    from slam.infrastructure.transformations_2d import Transformations2D
    from slam.infrastructure.defs import Pose2D
    


'''


'''


NODE_NAME = "rbpf_slam_node"

USE_DEBUGGER = False

# Pose names
POSE_ERR_TRUE_BEST_P_TOPIC = "pose_err_true_best_p"
POSE_ERR_TRUE_MEAN_P = "pose_err_true_maen_p"



@dataclass
class tf_poses:
    map_to_base_pose: Pose2D = (0.0, 0.0, 0.0)
    odom_to_base_pose: Pose2D = (0.0, 0.0, 0.0)
    map_to_odom_pose: Pose2D = (0.0, 0.0, 0.0)


@dataclass
class Colormap:
    col_val_unknown: int = -1
    col_val_occ: int = 100
    col_val_free: int = 0


@dataclass
class ROSParams:
    rbpf_input_topic: str
    map_topic: str
    true_pose_topic: str
    best_particle_pose_topic: str
    weighted_mean_particle_pose_topic: str
    map_tf_frame: str
    odom_tf_frame: str
    base_tf_frame: str
    laser_tf_frame: str
    input_queue_size: int
    tf_timeout_s: float
    max_filter_duration_s: float
    col_map: Colormap = field(default_factory=Colormap)



def debug_code():
    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()
    print("Debugger attached")



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


def _initialize_experiment_tag(exp_params: ExperimentParams) -> ExperimentParams:
    '''
    Initialize the experiment tag based on the parameters of the experiment. 
    '''
    # Keep this template aligned with optimize_rbpf_multiple_particles/tune_rbpf.py.
    sigma_meas = 0.06
    mm = exp_params.measurement_model_params
    motion = exp_params.motion_model_params
    scan_match = exp_params.scan_matcher_params
    particle = exp_params.particle_params

    exp_params.tag = (
        f"meas{sigma_meas}_nthf{exp_params.every_nth_scan_filter}_nmp{exp_params.every_nth_scan_map}_npart{particle.n_particles}_"
        f"smxy{motion.sigma_x}_smth{motion.sigma_theta}_cmf{motion.ctrl_motion_fac}_ctf{motion.ctrl_turn_fac}_"
        f"neff{exp_params.neff_threshold}_psig{exp_params.proposal_sigma_xy}_psth{exp_params.proposal_sigma_theta}_"
        f"nsdir{exp_params.proposal_n_samples}_mks{exp_params.meas_kernel_size}_"
        f"covss{exp_params.cov_std_scale}_covmsxy{exp_params.cov_max_std_xy}_covmsth{exp_params.cov_max_std_theta}_"
        f"minstdxy{exp_params.min_std_xy}_minstdth{exp_params.min_std_theta}_"
        f"boct{mm.occ_thresh}_bfr{mm.free_thresh}_buth{mm.unknown_ratio_thresh}_bkfr{mm.known_free_ratio_thresh}_"
        f"bsh{mm.sigma_hit}_bwh{mm.w_hit}_bws{mm.w_short}_bls{mm.lambda_short}_"
        f"bwm{mm.w_max}_bwr{mm.w_rand}_"
        f"bpun{mm.p_unknown}_bpoom{mm.p_out_of_map}_bpukf{mm.p_unexpected_known_free}_bppbm{mm.p_pred_below_min}_"
        f"bam{mm.alpha_meas}_bs{mm.beam_step}_"
        f"pa{exp_params.proposal_alpha}_pb{exp_params.proposal_beta}_surf{scan_match.surface_radius_m}_mfr{scan_match.min_free_ratio}"
    )

    return exp_params



def load_experiment_params(start_pose: Pose2D) -> ExperimentParams:
    '''
    Load the RBPF experiment configuration and add robot-specific values.
    '''
    config = rospy.get_param("~experiment")
    wheel_separation = load_wheel_separation()

    try:
        icp_config = dict(config["icp_params"])
        downsample_grid_size = icp_config.pop("downsample_grid_size")

        exp_params = ExperimentParams(
            measurement_model_params=BeamRangeFinderMeasModelParams(
                **config["measurement_model_params"]
            ),
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
                downssample_grid_size=downsample_grid_size,
                **icp_config,
            ),
            robot_params=RobotParams(
                wheel_separation=wheel_separation,
            ),
            scan_matcher_params=ScanMatcherParams(
                **config["scan_matcher_params"]
            ),
            particle_params=ParticleParams(
                start_pose=start_pose,
                **config["particle_params"],
            ),
            motion_model_params=MotionModelParams(
                wheel_separation=wheel_separation,
                **config["motion_model_params"],
            ),
            tag="",
            **config["rbpf_params"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid RBPF experiment configuration: {exc}"
        ) from exc

    return _initialize_experiment_tag(exp_params=exp_params)



def load_ros_node_params() -> ROSParams:
    '''Load ROS topics, frames, and runtime settings from configuration.'''
    config = rospy.get_param("~ros")

    try:
        topics = config["topics"]
        frames = config["frames"]
        runtime = config["runtime"]

        return ROSParams(
            rbpf_input_topic=topics["rbpf_input"],
            map_topic=topics["map"],
            true_pose_topic=topics["true_pose"],
            best_particle_pose_topic=topics["best_particle_pose"],
            weighted_mean_particle_pose_topic=(
                topics["weighted_mean_particle_pose"]
            ),
            map_tf_frame=frames["map"],
            odom_tf_frame=frames["odom"],
            base_tf_frame=frames["base"],
            laser_tf_frame=frames["laser"],
            input_queue_size=int(runtime["input_queue_size"]),
            tf_timeout_s=float(runtime["tf_timeout_s"]),
            max_filter_duration_s=float(
                runtime["max_filter_duration_s"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid RBPF ROS configuration: {exc}"
        ) from exc


def load_robot_start_pose() -> Pose2D:
    '''Load the robot start pose supplied by the launch file.'''
    robot_start_pose = (
        float(rospy.get_param("/spawn_x")),
        float(rospy.get_param("/spawn_y")),
        float(rospy.get_param("/spawn_yaw")),
    )
    rospy.loginfo("Loaded robot start pose: %s", robot_start_pose)
    return robot_start_pose



class RBPF_ROS_Node:
    def __init__(
        self,
        rbpf: RBPF,
        raw_odom_est: RawOdomEstimator,
        evaluator: RBPFEvaluator,
        exp_params: ExperimentParams,
        ros_params: ROSParams
    ):
        self.rbpf = rbpf
        self.raw_odom_est = raw_odom_est
        self.evaluator = evaluator
        self.filter_params = exp_params
        self.ros_params = ros_params

        # Define obj to locks threads
        self.lock = threading.Lock()

        # Initialize ROS publishers and subscribers
        # Init subscirber and Queue to store input data
        self.rbpf_input_queue = Queue(maxsize=self.ros_params.input_queue_size)
        self.rbpf_input_sub = rospy.Subscriber(
            name=self.ros_params.rbpf_input_topic,
            data_class=RBPFInput,
            callback=self._rbpf_input_cb,
            queue_size=5
        )

        # Def Publisher
        # Map publisher
        self.map_pub = rospy.Publisher(
            name=self.ros_params.map_topic,
            data_class=OccupancyGrid,
            queue_size=1,
            latch=True,
        )

        # Pose publisher
        self.pose_pub = {            
            self.ros_params.true_pose_topic: rospy.Publisher(
                name=self.ros_params.true_pose_topic,
                data_class=PoseStamped,
                queue_size=2
            ),
            self.ros_params.weighted_mean_particle_pose_topic: rospy.Publisher(
                name=self.ros_params.weighted_mean_particle_pose_topic,
                data_class=PoseStamped,
                queue_size=2
            ),
            self.ros_params.best_particle_pose_topic: rospy.Publisher(
                name=self.ros_params.best_particle_pose_topic,
                data_class=PoseStamped,
                queue_size=2
            ),                        
        }

        # Define member for topic data storage
        self.laser_scan = None
        self.dl = 0.0
        self.dr = 0.0
        self.pose = None

        # Initialize TF broadcaster
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # TODO: Adapt the RBPF filter such that it can handle tfs. Best would be if scan points would be transformed to base frame 
        # and filter takes this as input instead of the range, bearing Lists/arrays.As an alternative we could tf the range, bearing
        # data directly into the base frame. This way the rbpf input doesnt need to change.
        self.base_to_laser_pose_2d = self._tf_2D(
            src_frame=self.ros_params.laser_tf_frame,
            targ_frame=self.ros_params.base_tf_frame,
        )


    def _tf_2D(self, src_frame, targ_frame):
        '''
        Defined 2D transf from src_frame to targ_frame. Returns the tf as a 2D tuple (x, y, yaw) in the target frame.
        Raises
        ------
        RuntimeError
            If the tf cannot be found within the timeout period.

        Parameters
        ----------
        src_frame : str
            The source frame of the transformation
        targ_frame : str
            The target frame of the transformation

        Returns
        -------
        tuple
            A tuple (x, y, yaw) representing the 2D transformation from src_frame to targ_frame, where x and y 
            are the translation components and yaw is the rotation around the z-axis. 
        '''
        timeout_time = (
            rospy.Time.now()
            + rospy.Duration(self.ros_params.tf_timeout_s)
        )

        while not rospy.is_shutdown() and rospy.Time.now() < timeout_time:
            try:
                # Compute tf laser -> base
                tf = self.tf_buffer.lookup_transform(
                    target_frame=targ_frame,
                    source_frame=src_frame,
                    time=rospy.Time(0),
                    timeout=rospy.Duration(1.0)
                )

                # Transfer to 2d tf
                trans = tf.transform.translation
                rot = tf.transform.rotation

                # Transform rot
                (_, _, yaw) = euler_from_quaternion(
                    [rot.x, rot.y, rot.z, rot.w]
                )

                tf = (trans.x, trans.y, yaw)

                # Log successful tf 
                rospy.loginfo(
                    "Cached %s->%s offset (2D): x=%.3f, y=%.3f, yaw=%.3f rad",
                    src_frame,
                    targ_frame,
                    trans.x,
                    trans.y,
                    yaw,
                )

                return tf

            # Except failure while waiting
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                rospy.logwarn_throttle(
                    5.0,
                    f"Waiting for TF {self.ros_params.base_tf_frame} -> {self.ros_params.laser_tf_frame}",
                )

        # Throw error if timeout exceeded and no successful tf was found
        raise RuntimeError(
            f"Timeout of {self.ros_params.tf_timeout_s} exceeded for TF {self.ros_params.base_tf_frame} -> {self.ros_params.laser_tf_frame}"
        )
    

    @staticmethod
    def _pose_into_pose_stamped_msg(
        pose: Tuple[float, float, float],
        frame: str,
        stamp: Optional[rospy.Time] = None,
    ) -> PoseStamped:
        '''
        Convert a 2D pose (x, y, theta) into a ROS PoseStamped message with the specified frame and timestamp.

        Parameters
        ----------
        pose : Tuple[float, float, float]
            A tuple representing the 2D pose (x, y, theta) where theta is the orientation in radians.
        frame : str
            The frame ID for the PoseStamped message.
        stamp : Optional[rospy.Time], optional
            The timestamp for the PoseStamped message. If None, the current time will be used.
        
        Returns
        -------
        PoseStamped
            A ROS PoseStamped message containing the specified pose, frame, and timestamp.
        '''
        if pose is None or len(pose) != 3:
            raise ValueError("Pose must be a tuple of (x, y, theta)")
        if frame is None or not isinstance(frame, str):
            raise ValueError("Frame must be a valid string")

        msg = PoseStamped()
        msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
        msg.header.frame_id = frame
        msg.pose.position.x = pose[0]
        msg.pose.position.y = pose[1]
        msg.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        quat = quaternion_from_euler(0, 0, pose[2])
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]

        return msg


    @staticmethod
    def _map_into_occupancy_grid_msg(
        map_raveled: np.ndarray,
        timestamp: rospy.Time,
        frame_id: str,
        grid_res: float,
        width: int,
        height: int,
        origin_x: float,
        origin_y: float,
        orient_yaw: float=0.0,
    ):
        '''
        Creates a ROS OccupancyGrid message from the given flattened map and it's metadata.

        Parameters
        ----------
        map : np.ndarray
            A 1D numpy array representing the the map.
        timestamp : Union[rospy.Time, float]
            The timestamp for the OccupancyGrid message. Can be a rospy.Time object or a float representing seconds since epoch.
        frame_id : str
            The frame ID for the OccupancyGrid message.
        grid_res : float
            The resolution of the grid in meters per cell.
        width : int
            The width of the grid in number of cells.
        height : int
            The height of the grid in number of cells.
        origin_x : float
            The x-coordinate of the origin of the grid in the map frame.
        origin_y : float
            The y-coordinate of the origin of the grid in the map frame.
        orient_yaw : float, optional
            The yaw orientation of the grid in radians. Default is 0.0.
        '''
        
        # Init OGM message
        map_msg = OccupancyGrid()

        timestamp = timestamp if timestamp is not None else rospy.Time.now()
        
        # Add general info
        map_msg.header.stamp = timestamp
        map_msg.header.frame_id = frame_id

        # Add map 
        map_msg.data = map_raveled

        # Add map metadata
        map_msg.info.map_load_time = timestamp
        map_msg.info.resolution = grid_res
        map_msg.info.width = width
        map_msg.info.height = height

        map_msg.info.origin.position.x = origin_x
        map_msg.info.origin.position.y = origin_y
        map_msg.info.origin.position.z = 0.0
        
        quat = quaternion_from_euler(0, 0, orient_yaw)
        map_msg.info.origin.orientation.x = quat[0]
        map_msg.info.origin.orientation.y = quat[1]
        map_msg.info.origin.orientation.z = quat[2]
        map_msg.info.origin.orientation.w = quat[3]

        return map_msg


    @staticmethod
    def transform_laser_scan_to_measurement(laser_scan: LaserScan):
        '''
        Transform the sensor msgs LaserScan to a list of measurement's consisting of 
        (range, bearing) tuple.
        '''
        min_angle = laser_scan.angle_min
        angle_increment = laser_scan.angle_increment
        bearing = min_angle
        measurements = []
        
        # Transform LaserScan data
        for r in laser_scan.ranges:
            measurement = (r, bearing)
            measurements.append(measurement)
            bearing += angle_increment
            
        return measurements  


    def _rbpf_input_cb(self, msg: RBPFInput) -> None:
        # Store message into queue for processing in the main loop       
        try:
            self.rbpf_input_queue.put_nowait(msg)
        except Full:
            rospy.logerr_throttle(
                period=2.0,
                msg="\nRBPF input queue is full. The filter cannot process the incoming data fast enough!",
            )


    def _publish_pose(
        self,
        topic_key: str,
        pose: Tuple[float, float, float],
        frame_id: str,
        stamp: Optional[rospy.Time] = None
    ) -> None:
        msg = self._pose_into_pose_stamped_msg(
            pose=pose,
            frame=frame_id,
            stamp=stamp,
        )
        self.pose_pub[topic_key].publish(msg)


    def _publish_data(
        self,
        step_res: StepResult,      
        info: Dict,  
        timestamp: Optional[rospy.Time] = None,
    ):
        '''
        Methods that handles the overall publishing of data to ROS topics. 
        '''
        # Define one time stamp for all messages to be published 
        timestamp = timestamp if timestamp is not None else rospy.Time.now()

        # Publish tfs
        # Compute tfs
        tf_msgs = self._compute_tfs(
            step_res=step_res,
            timestamp=timestamp,
        )

        if tf_msgs:
            self.tf_broadcaster.sendTransform(tf_msgs)

        # Publish poses
        pose_data = [
            (
                self.ros_params.true_pose_topic,
                step_res.true_pose,
                self.ros_params.map_tf_frame,
                timestamp,
            ),
            (
                self.ros_params.weighted_mean_particle_pose_topic,
                step_res.weighted_mean_pose,
                self.ros_params.map_tf_frame,
                timestamp,
            ),
            (
                self.ros_params.best_particle_pose_topic,
                step_res.best_particle_pose,
                self.ros_params.map_tf_frame,
                timestamp,
            ),
        ]
        for topic_key, pose, frame_id, t_stamp in pose_data:
            if pose is not None:
                self._publish_pose(
                    topic_key=topic_key,
                    pose=pose,
                    frame_id=frame_id,
                    stamp=t_stamp,
                )

        # Publish map
        best_p_map: np.ndarray = info.get("best_particle_map", None)
        best_p_map_meta: Dict = info.get("best_particle_map_meta", None)

        if best_p_map is not None and best_p_map_meta is not None:
            # Convert log odds map to discretized color value map
            col_map = self.convert_log_odds_map(
                best_p_map=best_p_map,
                best_p_map_meta=best_p_map_meta
            )

            # Init OGM message
            map_msg = self._map_into_occupancy_grid_msg(
                map_raveled=col_map.ravel(order="C").tolist(),
                timestamp=timestamp if timestamp is not None else rospy.Time.now(),
                frame_id=self.ros_params.map_tf_frame,
                grid_res=float(best_p_map_meta.get("grid_resolution_m")),
                width=int(best_p_map_meta.get("number_of_cells_x")),
                height=int(best_p_map_meta.get("number_of_cells_y")),
                origin_x=float(- best_p_map_meta.get("shift_x")),
                origin_y=float(- best_p_map_meta.get("shift_y")),
                orient_yaw=0.0
            )

            # Publish map
            self.map_pub.publish(map_msg)
        else:
            rospy.logwarn(
                f"Best particle map or its metadata is not available. Skipping map publishing in step {step_res.step_idx}."
            )


    def _run_filter(
        self,
        laser_scan: LaserScan,
        dl: float, 
        dr: float,
    ):
        '''
        Runs the rbpf filter with the given input data. Transforms the given input data into the required format and calls 
        the rbpf filter's step method to precit the robot pose and estimate the map. 

        Parameters
        ----------
        laser_scan : LaserScan
            The laser scan data from the robot's laser scanner.
        dl : float
            The left wheel encoder delta (change in position) since the last update.
        dr : float
            The right wheel encoder delta (change in position) since the last update.
        '''
        # Transform measurements to (range, bearing) tuples
        measurements = self.transform_laser_scan_to_measurement(laser_scan)

        # Subsample and clean measuremnts for filter and map
        every_nth_scan_filter = self.filter_params.every_nth_scan_filter
        measurements_filter = (
            measurements[::every_nth_scan_filter] if every_nth_scan_filter > 1 else measurements
        )

        measurements_filter = [
            (r, b) for r, b in measurements_filter if np.isfinite(r)
        ]

        every_nth_scan_map = self.filter_params.every_nth_scan_map
        measurements_map = (
            measurements[::every_nth_scan_map] if every_nth_scan_map > 1 else measurements
        )

        # Compute raw odometry pose
        self.raw_odom_est.predict_pose(dl=dl, dr=dr)

        # Do rbpf update step
        self.rbpf.step(
            odom=(dl, dr),
            measurements_proposal=measurements_filter,
            measurements_map_update=measurements_map,
            proposal_sigma_xy=self.filter_params.proposal_sigma_xy,
            proposal_sigma_theta=self.filter_params.proposal_sigma_theta,
            proposal_n_samples=self.filter_params.proposal_n_samples,
            cov_std_scale=self.filter_params.cov_std_scale,
            cov_max_std_xy=self.filter_params.cov_max_std_xy,
            cov_max_std_theta=self.filter_params.cov_max_std_theta,
            min_std_xy=self.filter_params.min_std_xy,
            min_std_theta=self.filter_params.min_std_theta,
        )


    def convert_log_odds_map(
        self,
        best_p_map: np.ndarray,
        best_p_map_meta: Dict,
    ) -> np.ndarray:
        if best_p_map is None or best_p_map_meta is None:
            rospy.logwarn("No best particle map available to convert to LogOddsMap message.")
            return None

        discretized_map = OGM.discretize_map(
            ogm=best_p_map,
            occ_thres=self.filter_params.measurement_model_params.occ_thresh,
            free_thres=self.filter_params.measurement_model_params.free_thresh,
            col_val_unknown=self.ros_params.col_map.col_val_unknown,
            col_val_free=self.ros_params.col_map.col_val_free,
            col_val_occ=self.ros_params.col_map.col_val_occ,
        )
        return discretized_map


    @staticmethod
    def __read_rbpf_info(info: Dict, key: str):
        '''
        Cheks whether the given info obj is indeed and dict and if the key is a str. If not raises a ValueError. 
        Alos checks if the given key is in the info dict.

        Parameters
        ----------
        info : Dict
            The info dictionary from the RBPF filter containing various information about the current step.
        key : str
            The key to look for in the info dictionary.
        
        Returns
        -------
        value: Any
            The value associated with the given key in the info dictionary. If the key does not exist, raises a KeyError.
        '''
        # Check if info is a dictionary and key is a string
        if not isinstance(info, dict):
            raise ValueError("RBPF info must be a dictionary.")

        if not isinstance(key, str):
            raise ValueError("RBPF info key must be a string.")

        # Check if the key exists in the info dictionary
        if key not in info.keys():
            raise KeyError(f"Key '{key}' not found in RBPF info dictionary. Keys are:\n{list(info.keys())}")
        
        value = info.get(key, None)
        
        return value


    def _evaluate_run(
        self,
        step_time: float,
        step_duration: float,
        filter_duration: float,
        msg_queue_size: int,
        true_pose: Optional[Pose2DMsg] = None
    ):
        # Evaluate step results
        step_res = None
        info = self.rbpf.get_step_info()

        if info is None:
            raise ValueError("RBPF info is None. Cannot evaluate run without step information.")

        # Extract data 
        step_idx = self.__read_rbpf_info(info, "step") 
        scan_match_failed = self.__read_rbpf_info(info, "scan_match_failed_any")
        scan_match_fallback_failed = self.__read_rbpf_info(info, "scan_match_fallback_failed_any")
        particle_poses = self.__read_rbpf_info(info, "particle_poses")
        particle_weights = self.__read_rbpf_info(info, "particle_weights")
        particle_poses_before_resampling = self.__read_rbpf_info(info, "particle_poses_before_resampling")
        particle_weights_before_resampling = self.__read_rbpf_info(info, "particle_weights_before_resampling")
        neff = self.__read_rbpf_info(info, "neff")
        resampled_indices = self.__read_rbpf_info(info, "resampled_indices")

        # Evaluate step results
        step_res = self.evaluator.evaluate_step(
            step_idx=step_idx,
            t=step_time,
            step_duration=step_duration,
            filter_duration=filter_duration,
            msg_queue_size=msg_queue_size,
            scan_match_failed=scan_match_failed,
            scan_match_fallback_failed=scan_match_fallback_failed,
            raw_odom_pose=self.raw_odom_est.get_pose(),
            particle_poses=particle_poses,
            particle_weights=particle_weights,
            particle_poses_before_resampling=particle_poses_before_resampling,
            particle_weights_before_resampling=particle_weights_before_resampling,
            neff=neff,
            particle_inherit_indices=resampled_indices,
            true_pose=true_pose,
        )

        return step_res, info

    
    def display_information(
        self,
        step_res: StepResult,
        total_step_time: float
    ):
        # rospy.logwarn(f"Total step time is: {total_step_time} s")
        if step_res is None:
            rospy.logwarn(f"Step result not intialized -> No information to display!")
            return

        if step_res.t_filter_duration is not None:
            if (step_res.t_filter_duration > self.ros_params.max_filter_duration_s):
                t_filter_duration_ms = step_res.t_filter_duration * 1000.0
                max_filter_duration_ms = (
                    self.ros_params.max_filter_duration_s * 1000.0
                )
                rospy.logwarn(
                    f"\nFilter duration exceeded given threshold."
                    f"Filter duration: {t_filter_duration_ms:.4f} ms > {max_filter_duration_ms:.4f} ms."
                    f"In step {step_res.step_idx}."
                )
                rospy.loginfo(f"Complete Step {step_res.step_idx} took {step_res.t_step_duration:.4f} seconds.")

        if step_res.resampling:
            rospy.loginfo(f"Resampling took place in step {step_res.step_idx}")

        if step_res.scan_match_failed:
            rospy.logwarn(f"Scan Matching failed in step {step_res.step_idx}!")

        if step_res.scan_match_fallback_failed:
            rospy.logwarn(f"Scan Matching fallback failed in step {step_res.step_idx}!")


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

    
    def _compute_tfs(
        self,
        step_res: StepResult,
        timestamp: rospy.Time,
    ) -> List[TransformStamped]:
        """
        Create the currently available dynamic transforms.

        Returns
        -------
        List[TransformStamped]
            Empty list:
                No transform can currently be published.

            One transform:
                Only odom -> base is available.

            Two transforms:
                map -> odom and odom -> base are available.
        """
        # List to add transforms msgs to
        transforms = []

        raw_odom_pose = step_res.raw_odom_pose
        best_particle_pose = step_res.best_particle_pose

        # Skip tf computation if raw ododm is missing
        if raw_odom_pose is None:
            rospy.logwarn_throttle(
                2.0,
                "Raw odometry pose is unavailable. Skipping TF publication.",
            )
            return transforms

        # Transfer raw odometry pose into tf msg
        odom_to_base_tf = self._pose_into_transform_stamped_msg(
            pose=raw_odom_pose,
            parent_frame=self.ros_params.odom_tf_frame,
            child_frame=self.ros_params.base_tf_frame,
            timestamp=timestamp,
        )

        transforms.append(odom_to_base_tf)

        # Display warning when best particle pose unavailable -> can't compute map -> odom tf
        if best_particle_pose is None:
            rospy.logwarn_throttle(
                2.0,
                "Best-particle pose is unavailable. "
                "Publishing only odom -> base_link.",
            )
            return transforms

        # Compute map -> odom tf
        base_to_odom_pose = Transformations2D.inverse(
            raw_odom_pose
        )

        map_to_odom_pose = Transformations2D.compose(
            first_transform=best_particle_pose,
            second_transform=base_to_odom_pose,
        )

        # Transfer map_odom into tf msg
        map_to_odom_tf = self._pose_into_transform_stamped_msg(
            pose=map_to_odom_pose,
            parent_frame=self.ros_params.map_tf_frame,
            child_frame=self.ros_params.odom_tf_frame,
            timestamp=timestamp,
        )

        transforms.append(map_to_odom_tf)

        return transforms
            

    def exe(self) -> None:
        '''
        Main loop that executes the algorithm.
        '''
        while not rospy.is_shutdown():
            # Extract new data from queue
            try:
                msg: RBPFInput = self.rbpf_input_queue.get(timeout=0.1)
                msg_queue_size = self.rbpf_input_queue.qsize()
            except Empty:    
                continue

            try:
                # Measure start time
                start_time = time.perf_counter()

                # Extract data from message
                laser_scan = msg.laser_scan
                dl = msg.wheel_encoder.left
                dr = msg.wheel_encoder.right

                true_pose = msg.true_pose if hasattr(msg, "true_pose") else None

                step_timestamp = msg.header.stamp 
                step_time = step_timestamp.to_sec()

                # Run rbpf filter
                start_time_filter = time.perf_counter()
                self._run_filter(
                    laser_scan=laser_scan,
                    dl=dl,
                    dr=dr,
                )

                filter_duration = time.perf_counter() - start_time_filter
                step_duration = time.perf_counter() - start_time

                # Evaluate step results
                step_res, info = self._evaluate_run(
                    step_time=step_time,
                    step_duration=step_duration,
                    filter_duration=filter_duration,
                    msg_queue_size=msg_queue_size,
                    true_pose=true_pose,
                )

                if step_res is None or info is None:
                    rospy.logwarn("No RBPF step information available. Publishing data will be skipped this step.")
                    continue

                # Publish data
                self._publish_data(
                    step_res=step_res,
                    info=info,
                    timestamp=step_timestamp,
                )

                total_step_time = time.perf_counter() - start_time

                self.display_information(step_res=step_res, total_step_time=total_step_time)
                                
            except Exception:
                rospy.logerr(
                    "Error processing RBPF input message:\n%s",
                    traceback.format_exc(),
                )

            finally:
                self.rbpf_input_queue.task_done()
                
                        

def init():
    # Init ros node
    rospy.init_node(NODE_NAME)

    # Load configuration and values supplied by the robot launch
    config_name = rospy.get_param("~config_name")
    robot_start_pose = load_robot_start_pose()
    ros_params = load_ros_node_params()
    
    # Define experiment parameters
    exp_params = load_experiment_params(start_pose=robot_start_pose)
    rospy.loginfo("Loaded RBPF configuration: %s", config_name)

    # Init Robot Estimator classes
    rbpf_factory = RBPFFactory()
    rbpf = rbpf_factory.create(
        scan_match_fac=ScanMatchFactory(),
        particle_params=exp_params.particle_params,
        occ_param=exp_params.occupancy_params,
        sens_params=exp_params.sensor_params,
        map_param=exp_params.map_param,
        icp_params=exp_params.icp_params,
        robot_params=exp_params.robot_params,
        scan_matcher_params=exp_params.scan_matcher_params,
        motion_model_params=exp_params.motion_model_params,
        measurement_model_params=exp_params.measurement_model_params,
        neff_threshold=exp_params.neff_threshold,
    )

    raw_odom_est = RawOdomEstimator(
        motion_model=MotionModel(
            sigma_x=exp_params.motion_model_params.sigma_x,
            sigma_y=exp_params.motion_model_params.sigma_y,
            sigma_theta=exp_params.motion_model_params.sigma_theta,
            wheel_separation=exp_params.motion_model_params.wheel_separation,
            ctrl_motion_fac=exp_params.motion_model_params.ctrl_motion_fac,
            ctrl_turn_fac=exp_params.motion_model_params.ctrl_turn_fac,
        ),
        start_pose=robot_start_pose
    )

    # Init evaluator
    evaluator = RBPFEvaluator()

    return rbpf, raw_odom_est, evaluator, exp_params, ros_params
    


def main():
    if USE_DEBUGGER:
        debug_code()

    # Init RBPF filter
    rbpf, raw_odom_est, evaluator, exp_params, ros_params = init()

    rbpf_ros_node = RBPF_ROS_Node(
        rbpf=rbpf,
        raw_odom_est=raw_odom_est,
        evaluator=evaluator,
        exp_params=exp_params,
        ros_params=ros_params,
    )

    rbpf_ros_node.exe()


if __name__ == "__main__":
    main()