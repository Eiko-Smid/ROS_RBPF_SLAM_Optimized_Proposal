#!/usr/bin/env python3
import debugpy

from typing import List, Tuple, Optional, Dict, Any, List, Union

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
from rbpf_slam.msg import LogOddsMap
from geometry_msgs.msg import PoseStamped

from dataclasses import dataclass
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
    from rbpf_slam.src.slam.rbpf.motion_model import MotionModel
    from rbpf_slam.src.slam.optimize_rbpf_multiple_particles.playback_defs import ExperimentParams, PlaybackData
    from rbpf_slam.src.slam.infrastructure.playback_recorder import PlaybackRecorder, build_metadata
    
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

    from slam.rbpf.motion_model import MotionModel

    from slam.optimize_rbpf_multiple_particles.playback_defs import ExperimentParams, PlaybackData
    from slam.infrastructure.playback_recorder import PlaybackRecorder, build_metadata


NODE_NAME = "rbpf_slam_node"

'''

TODO: Load start pose
    Status: Done


TODO: Track robot pose and publish
    - Track some poses like:
        - true pose (if available)
        - best particle pose
        - weighted mean particle pose
    - Publish these poses to ROS topics

    Status: Done

TODO: Track info 
    - Track the following information and display even based
        
        input_queue_size
        Neff
        resampling_performed
        scan_matching_failed
    
    - Track every n steps
        processing_time

    
    Status: Done

    
TODO: Publish best particle map
    - Easiest v1 would be to simply publish the map of the best particle. Maybe count after how many updates shat should be done
    - Doing it too often results in performance issues, doing it too infrequently results in low FPS map updates that don't look nice

    Status: Done
    

TODO: Adapt Map transformation node
    - Currently the node publishes with its update rate
    - This is unnessesary because we only need to publish when we actually received a new msg in the msg queue
    
    Changes:
        - Add msg queue to Node
        - Delete the update rate and do it as we do it here !

    Status: todo


TODO: Add raw odom
    - Compute raw odom
    - Add to step data 
    - before implementing check for tfs in ros cause ros needs raw odom as far as i know.
    
    Status: Done


    
TODO: Read wheel separation from robot description or ROS parameter server.
    
    Status: todo

'''

USE_DEBUGGER = False

MIN_SENSOR_RANGE = 0.1
MAX_SENSOR_RANGE = 10.0

# Pose names
TRUE_POSE_TOPIC = "true_pose"
BEST_P_POSE = "best_particle_pose"
WEIGHTED_MEAN_P_POSE = "weighted_mean_particle_pose"
POSE_ERR_TRUE_BEST_P_TOPIC = "pose_err_true_best_p"
POSE_ERR_TRUE_MEAN_P = "pose_err_true_maen_p"


Pose2D = Tuple[float, float, float]


@dataclass
class ROSParams:
    # Update rate
    update_rate: int = 2
    
    # Link states topic and params
    link_state_topic: str = "/gazebo/link_states"
    link_state_name: str = "robot_vacuum_cleaner::base_link"

    # RBPF input topic
    rbpf_input_topic: str = "rbpf/input"
    input_queue_size = 10
    # log odds map topic
    map_topic: str = "log_odds_map"    

    # TFs
    map_tf_frame: str = "map"
    odom_tf_frame: str = "odom_link"
    base_tf_frame: str = "base_link"
    laser_tf_frame: str = "laser_scanner_link"
    tf_timeout_s = rospy.Duration(10.0)


@dataclass
class StepResult:
    # General metrics
    step_idx: Optional[int] = None
    t: Optional[float] = None
    t_step_duration: Optional[float] = None
    msg_queue_size: Optional[int] = None

    # Ground truth and odom
    true_pose: Optional[np.ndarray] = None
    raw_odom_pose: Optional[np.ndarray] = None

    # Scan matcher info
    scan_match_failed: Optional[bool] = None
    scan_match_failed_fallback: Optional[bool] = None

    # Raw odom pose
    raw_odom_pose: Optional[np.ndarray] = None

    # Particle poses before and after resampling
    particle_poses: Optional[np.ndarray] = None
    particle_weights: Optional[np.ndarray] = None
    particle_poses_before_resampling: Optional[np.ndarray] = None
    particle_weights_before_resampling: Optional[np.ndarray] = None

    # Weighted mean pose
    weighted_mean_pose: Optional[np.ndarray] = None

    # Best particle pose and weight
    max_weight_idx: Optional[int] = None
    best_particle_pose: Optional[np.ndarray] = None
    best_particle_weight: Optional[float] = None

    neff: Optional[float] = None
    resampling: Optional[bool] = None


def debug_code():
    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()
    print("Debugger attached")



def compute_wheel_separation():
    '''
    Compute the wheel separation based on the robot's chassis and wheel dimensions.
    '''
    h_chassis = 0.15
    dist_chassis_to_ground = h_chassis / 5
    r_wheel = h_chassis / 2 + dist_chassis_to_ground
    w_wheel = 0.3 * r_wheel
    r_chassis = 0.25
    wheel_separation = 2 * r_chassis + w_wheel

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



def def_exp_params(start_pose):
    '''
    Returns an instance of the initialized Experiment Parameters for the RBPF filter.
    '''
    wheel_separation = compute_wheel_separation()

    measurement_model_params = BeamRangeFinderMeasModelParams(
        occ_thresh=1.4,
        free_thresh=-1.4,
        unknown_ratio_thresh=0.3,
        known_free_ratio_thresh=0.7,
        sigma_hit=0.07,
        w_hit=0.5,
        w_short=0.3,
        lambda_short=0.20,
        w_max=0.10,
        w_rand=0.10,
        p_unknown=0.10,
        p_out_of_map=0.15,
        p_unexpected_known_free=0.00,
        p_pred_below_min=0.02,
        alpha_meas=0.075,
        beam_step=2,
        eps=1e-12,
    )

    occupancy_params = OccupancyParams(
        prior_probability=0.5,
        min_distance_to_border=10.0,
        increasing_probability=0.85,
        decreasing_probability=0.15,
        min_log_odds=-5.0,
        max_log_odds=5.0,
    )
    sensor_params = SensorParams(
        min_sensor_range=MIN_SENSOR_RANGE,
        max_sensor_range=MAX_SENSOR_RANGE,
    )
    map_param = MapParameter(
        map_width=10.0,
        map_height=10.0,
        grid_resolution_m=0.05,
    )
    icp_params = ICPParams(
        max_n_points=1200,
        downssample_grid_size=0.1,
        max_correspondence_distance=0.4,
        neighbors_pca=6,
        max_iterations=5,
        epsilon_rel=1e-3,
        no_improvement_limit=3,
        min_error=5e-4,
        min_dtrans=1e-3,
        min_drot=1e-2,
        min_points=20,
        min_corresp=25,
        min_hessian_rank=3,
        max_hessian_condition=1e8,
        max_translation_jump=0.7,
        max_rotation_jump=np.deg2rad(45.0),
        max_acceptable_mean_error=0.15,
    )
    robot_params = RobotParams(
        wheel_separation=wheel_separation,
    )
    scan_matcher_params = ScanMatcherParams(
        occ_thres=1.4,
        delta_r=0.6,
        surface_radius_m=0.2,
        min_free_ratio=0.4,
    )
    particle_params = ParticleParams(
        n_particles=20,
        start_pose=start_pose,
    )
    motion_model_params = MotionModelParams(
        sigma_x=0.12,
        sigma_y=0.12,
        sigma_theta=0.11,
        wheel_separation=wheel_separation,
        ctrl_motion_fac=0.1,
        ctrl_turn_fac=0.15,
    )

    exp_params = ExperimentParams(
        occupancy_params=occupancy_params,
        sensor_params=sensor_params,
        map_param=map_param,
        icp_params=icp_params,
        robot_params=robot_params,
        scan_matcher_params=scan_matcher_params,
        particle_params=particle_params,
        motion_model_params=motion_model_params,
        measurement_model_params=measurement_model_params,
        every_nth_scan_filter=2,
        every_nth_scan_map=2,
        neff_threshold=6.0,
        proposal_sigma_xy=0.06,
        proposal_sigma_theta=np.deg2rad(1.432),
        proposal_n_samples=3,
        cov_std_scale=0.5,
        cov_max_std_xy=1.0,
        cov_max_std_theta=np.deg2rad(10.0),
        min_std_xy=0.0,
        min_std_theta=np.deg2rad(0.0),
        meas_kernel_size=1,
        gaussian_sigma=0.05,
        proposal_alpha=1.0,
        proposal_beta=1.0,
        measurement_noise_stddev=0.03,
        used_meas_model="LaserRangeFinderModel",
        tag="",
    )

    return _initialize_experiment_tag(exp_params=exp_params)



def load_ros_params():
    # Get motion error params
    motion_error_factor = rospy.get_param(
        "/motion_error_factor"
    )
    turn_error_factor = rospy.get_param(
        "/turn_error_factor"
    )

    # Get lidar params
    laser_range_resolution = rospy.get_param("/laser_range_resolution")
    laser_noise_type = rospy.get_param("/laser_noise_type")
    laser_noise_mean = rospy.get_param("/laser_noise_mean")
    laser_noise_stddv = rospy.get_param("/laser_noise_stddv")
    
    # Get robot spawn pose
    spawn_x = rospy.get_param("/spawn_x")
    spawn_y = rospy.get_param("/spawn_y")
    spawn_yaw = rospy.get_param("/spawn_yaw")
    robot_start_pose = (spawn_x, spawn_y, spawn_yaw)

    # Print loaded parameters
    rospy.loginfo(f"\n\nDisplay loaded parameters:")
    rospy.loginfo(f"Loaded motion error factor: {motion_error_factor}")
    rospy.loginfo(f"Loaded turn error factor: {turn_error_factor}")
    rospy.loginfo(f"Loaded laser range resolution: {laser_range_resolution}")
    rospy.loginfo(f"Loaded laser noise type: {laser_noise_type}")
    rospy.loginfo(f"Loaded laser noise mean: {laser_noise_mean}")
    rospy.loginfo(f"Loaded laser noise stddv: {laser_noise_stddv}")
    rospy.loginfo(f"Loaded robot spawn pose: {robot_start_pose}\n\n")

    return robot_start_pose



class RawOdomEstimator:
    '''
    RawOdomEstimator estimates the robot's pose based on raw odometry data. Get's a motion model and a start pose
    and then can be used to predict the robot's pose. 
    '''
    DEFAULT_START_POSE: Pose2D = (0.0, 0.0, 0.0)

    def __init__(
        self,
        motion_model: MotionModel,
        start_pose: Optional[Pose2D] = None
    ):
        self.motion_model = motion_model
        self.pose = start_pose if start_pose is not None else self.DEFAULT_START_POSE


    def predict_pose(self, dl: float, dr: float) -> Pose2D:
        '''
        Predicts the robot's pose based on the given odometry (dl, dr).

        Parameters
        ----------
        dl: float
            The distance traveled by the left wheel since the last update.
        dr: float
            The distance traveled by the right wheel since the last update.
        
        Returns
        -------
        Pose2D
            The predicted pose of the robot (x, y, theta).
        '''
        self.pose = self.motion_model.predict_pose(self.pose, dl, dr)
        return self.pose


    def get_pose(self) -> Pose2D:
        '''
        Returns the current pose of the robot.

        Returns
        -------
        Pose2D
            The current pose of the robot (x, y, theta).
        '''
        return self.pose
    

    def reset(self, start_pose: Optional[Pose2D] = None):
        '''
        Resets the robot's pose to the given start pose or to the default start pose if no start pose has been given. 

        Parameters
        ----------
        start_pose: Optional[Pose2D]
            The pose to reset the robot to. If None, the default start pose will be used.
        '''
        if start_pose is not None:
            self.pose = start_pose
        else: 
            self.pose = self.DEFAULT_START_POSE



class RBPFEvaluator:
    @staticmethod
    def _finite_values(values: List[Optional[float]]) -> np.ndarray:
        '''
        Converts the given list to a numpy array. Filters all non-finite values (inf, -inf, nan) and returns 
        the numpy array.
        '''
        arr = np.asarray(values, dtype=float)        
        return arr[np.isfinite(arr)]


    @staticmethod
    def _has_valid_metric_array(values: Optional[np.ndarray], min_size: int = 1) -> bool:
        """
        Returns True only if values is a numpy array with at least min_size finite entries.
        """
        if values is None or not isinstance(values, np.ndarray):
            return False
        if values.size < min_size:
            return False
        return bool(np.all(np.isfinite(values)))


    @staticmethod
    def _safe_mean(values: Optional[np.ndarray], min_size: int = 1, default: float = float("nan")) -> float:
        """
        Computes the mean only when a valid timing array is available; otherwise returns default.
        """
        if not RBPFEvaluator._has_valid_metric_array(values, min_size=min_size):
            return default
        return float(np.mean(values))

    
    def _pose_to_np_array(self, pose: Pose2D) -> np.ndarray:
        """
        Converts a pose tuple to a numpy array.
        """
        if pose is None:
            return None

        if hasattr(pose, "x") and hasattr(pose, "y") and hasattr(pose, "theta"):
            return np.array([pose.x, pose.y, pose.theta], dtype=np.float64)

        if isinstance(pose, (tuple, list, np.ndarray)) and len(pose) >= 3:
            return np.array(pose[:3], dtype=np.float64)

        raise TypeError(f"[Evaluator mp] Unsupported pose format: {type(pose)}")


    def _weight_to_np_array(self, weights: List[float]) -> np.ndarray:
        """
        Converts a list of weights to a numpy array.
        """
        if weights is None:
            return None

        return np.array(weights, dtype=np.float64)
        

    def _poses_to_np_array(self, poses: List[Pose2D]) -> np.ndarray:
        """
        Converts a list of pose tuples to a numpy array.
        """
        if poses is None:
            return None

        return np.array([self._pose_to_np_array(p) for p in poses], dtype=np.float64)


    @staticmethod
    def angle_diff(a: float, b: float) -> float:
        """
        Returns wrapped angular difference in [-pi, pi].
        """
        return np.arctan2(np.sin(a - b), np.cos(a - b))


    @staticmethod
    def translation_error(p1: Pose2D, p2: Pose2D) -> float:
        """
        Euclidean translation error in the x-y plane.
        """
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


    @staticmethod
    def weighted_mean_position(particle_poses: np.ndarray, particle_weights: np.ndarray) -> np.ndarray:
        '''
        Gets the particle poses and the corresponding weights and computes the weighted mean pose of all particles. 
        The x, y position is computed by weighting the position by its weights and then computing the mean. 
        
        The weighted mean orientation is computed by computing the weighted mean of the sin and cos of the angles. 
        This ensures that the mean orientation is computed correctly even if the angles wrap around 2*pi.
        '''
        # Convert to numpy array
        particle_poses = np.asarray(particle_poses, dtype=float)
        particle_weights = np.asarray(particle_weights, dtype=float)

        # Check dimensions, shape and finite values
        if particle_poses.ndim != 2 or particle_poses.shape[1] != 3:
            raise ValueError(
                f"Particle poses must have shape (N, 3) got {particle_poses.shape}"
            )
        
        if particle_weights.ndim != 1 or particle_weights.shape[0] != particle_poses.shape[0]:
            raise ValueError(
                f"Particle weights must have shape (N, ), where N is number of given poses"
                f"Number of given poses: {particle_poses.shape[0]} != number of weights: {particle_weights.shape[0]}"
            )
        
        if not np.all(np.isfinite(particle_poses)):
            raise ValueError(
                f"Particle poses contain non-finite values: {particle_poses}"
            )

        if not np.all(np.isfinite(particle_weights)):
            raise ValueError(
                f"Particle weights contain non-finite values: {particle_weights}"
            )

        # Normalize weights
        particle_weights = RBPFEvaluator.norm_weights(particle_weights)

        # Compute weighted position
        weighted_position = np.average(particle_poses[:, :2], axis=0, weights=particle_weights,)

        # Compute weighted orientation
        weighted_mean_rot = np.arctan2(
            np.sum(particle_weights * np.sin(particle_poses[:, 2])),
            np.sum(particle_weights * np.cos(particle_poses[:, 2]))
        )
        
        weighted_mean_poses = np.array([weighted_position[0], weighted_position[1], weighted_mean_rot])
        return weighted_mean_poses


    def evaluate_step(
        self,

        step_idx: Optional[int]=None,
        t: Optional[float]=None,
        step_duration: Optional[float]=None,
        msg_queue_size: Optional[int]=None,

        scan_match_failed: Optional[bool]=None,
        scan_match_fallback_failed: Optional[bool]=None,

        raw_odom_pose: Optional[Pose2D]=None,
        particle_poses: List[Pose2D]=None,
        particle_weights: List[float]=None,

        particle_poses_before_resampling: List[Pose2D]=None,
        particle_weights_before_resampling: List[float]=None,

        neff: Optional[float]=None,
        particle_inherit_indices: Optional[List[int]]=None,

        true_pose: Optional[Pose2D] = None,
    ) -> StepResult:        
        # Convert input data to numpy arrays
        particle_poses = self._poses_to_np_array(particle_poses)
        particle_weights = self._weight_to_np_array(particle_weights)
        particle_poses_before_resampling = self._poses_to_np_array(particle_poses_before_resampling)
        particle_weights_before_resampling = self._weight_to_np_array(particle_weights_before_resampling)

        # Convert optional input data to numpy arrays
        true_pose = self._pose_to_np_array(true_pose) if true_pose is not None else None
        raw_odom_pose = self._pose_to_np_array(raw_odom_pose) if raw_odom_pose is not None else None

        # Init step results with None values
        step_res = StepResult()

        # Add general info to step
        step_res.step_idx = step_idx
        step_res.t = t
        step_res.t_step_duration = step_duration
        step_res.msg_queue_size = msg_queue_size

        # Add scan matcher info
        step_res.scan_match_failed = scan_match_failed
        step_res.scan_match_failed_fallback = scan_match_fallback_failed

        # Add particle poses
        step_res.particle_poses = particle_poses
        step_res.particle_weights = particle_weights
        step_res.particle_poses_before_resampling = particle_poses_before_resampling
        step_res.particle_weights_before_resampling = particle_weights_before_resampling
        step_res.true_pose = true_pose
        step_res.raw_odom_pose = raw_odom_pose

        # Compute weighted mean pose
        if particle_poses is not None and particle_weights is not None:
            step_res.weighted_mean_pose = self.weighted_mean_position(
                particle_poses=particle_poses,
                particle_weights=particle_weights
            )

        # Compute best particle pose
        if particle_poses_before_resampling is not None and particle_weights_before_resampling is not None:
            step_res.max_weight_idx = np.argmax(particle_weights_before_resampling)
            step_res.best_particle_pose = particle_poses_before_resampling[step_res.max_weight_idx]
            step_res.best_particle_weight = particle_weights_before_resampling[step_res.max_weight_idx]

        # Resampling info
        step_res.neff = neff
        step_res.resampling = particle_inherit_indices is not None 
        
        return step_res



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
            data_class=LogOddsMap,
            queue_size=1
        )

        # Pose publisher
        self.pose_pub = {            
            TRUE_POSE_TOPIC: rospy.Publisher(
                name=TRUE_POSE_TOPIC,
                data_class=Pose,
                queue_size=2
            ),
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
            src_frame=self.ros_params.base_tf_frame,
            targ_frame=self.ros_params.laser_tf_frame
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
        timeout_time = rospy.Time.now() + self.ros_params.tf_timeout_s

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
        for range in laser_scan.ranges:
            measurement = (range, bearing)
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


    def _publish_map(self):
        '''
        Publishes the best particle's map to the ROS topic.
        '''
        best_particle_map = self.rbpf.get_best_particle_map()
        if best_particle_map is not None:
            log_odds_map_msg = LogOddsMap()
            log_odds_map_msg.header.stamp = rospy.Time.now()
            log_odds_map_msg.header.frame_id = self.ros_params.map_tf_frame
            log_odds_map_msg.width = best_particle_map.width
            log_odds_map_msg.height = best_particle_map.height
            log_odds_map_msg.resolution = best_particle_map.resolution
            log_odds_map_msg.log_odds_data = best_particle_map.log_odds.flatten().tolist()

            self.map_pub.publish(log_odds_map_msg)


    def _publish_data(
        self,
        step_res: StepResult,      
        info: Dict,  
    ):
        '''
        Methods that handles the overall publishing of data to ROS topics. This
        '''
        # Publish poses
        timestamp = step_res.t if step_res.t is not None else rospy.Time.now()
        pose_data = [
            (TRUE_POSE_TOPIC, step_res.true_pose, self.ros_params.map_tf_frame, timestamp),
            (WEIGHTED_MEAN_P_POSE, step_res.weighted_mean_pose, self.ros_params.map_tf_frame, timestamp),
            (BEST_P_POSE, step_res.best_particle_pose, self.ros_params.map_tf_frame, timestamp)
        ]
        for topic_key, pose, frame_id, stamp in pose_data:
            self._publish_pose(
                topic_key=topic_key,
                pose=pose,
                frame_id=frame_id,
                stamp=stamp,
            )

        # Publish map
        best_p_map: np.ndarray = info.get("best_particle_map", None)
        best_p_map_meta: Dict = info.get("best_particle_map_meta", None)

        if best_p_map is not None:
            # Init log odds message
            map_msg = LogOddsMap()
            map_msg.header.stamp = timestamp if timestamp is not None else rospy.Time.now()
            map_msg.header.frame_id = self.ros_params.map_tf_frame
            map_msg.data = best_p_map.ravel(order="C").tolist()

            map_msg.info.resolution = best_p_map_meta.get("grid_resolution_m")
            map_msg.info.width = best_p_map_meta.get("number_of_cells_x")
            map_msg.info.height = best_p_map_meta.get("number_of_cells_y")

            map_msg.info.origin.position.x = - best_p_map_meta.get("shift_x")
            map_msg.info.origin.position.y = - best_p_map_meta.get("shift_y")
            map_msg.info.origin.position.z = 0.0

            map_msg.info.origin.orientation.x = 0.0
            map_msg.info.origin.orientation.y = 0.0
            map_msg.info.origin.orientation.z = 0.0
            map_msg.info.origin.orientation.w = 1.0

            # Publish map
            self.map_pub.publish(map_msg)


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
        self.rbpf.step_range_finder_model(
            odom=(dl, dr),
            measurements_proposal=measurements_filter,
            measurements_map=measurements_map,
            proposal_sigma_xy=self.filter_params.proposal_sigma_xy,
            proposal_sigma_theta=self.filter_params.proposal_sigma_theta,
            proposal_n_samples=self.filter_params.proposal_n_samples,
            cov_std_scale=self.filter_params.cov_std_scale,
            cov_max_std_xy=self.filter_params.cov_max_std_xy,
            cov_max_std_theta=self.filter_params.cov_max_std_theta,
            min_std_xy=self.filter_params.min_std_xy,
            min_std_theta=self.filter_params.min_std_theta,
        )


    def _evaluate_run(
        self,
        step_time: float,
        step_duration: float,
        msg_queue_size: int
    ):
        # Evaluate step results
        step_res = None
        info = self.rbpf.get_step_info()

        if info is not None:
            step_res = self.evaluator.evaluate_step(
                step_idx=info.get("step", None),
                t=step_time,
                step_duration=step_duration,
                msg_queue_size=msg_queue_size,
                scan_match_failed=info.get("scan_match_failed", None),
                scan_match_fallback_failed=info.get("scan_match_fallback_failed", None),
                raw_odom_pose=self.raw_odom_est.get_pose(),
                particle_poses=info.get("particle_poses", None),
                particle_weights=info.get("particle_weights", None),
                particle_poses_before_resampling=info.get("particle_poses_before_resampling", None),
                particle_weights_before_resampling=info.get("particle_weights_before_resampling", None),
                neff=info.get("neff", None),
                particle_inherit_indices = info.get("resampled_indices", None),
                true_pose=info.get("true_pose", None),
            )

        return step_res, info


    @staticmethod
    def display_information(step_res: StepResult):
        if step_res is None:
            rospy.logwarn(f"Step result not intialized -> No information to display!")

        if step_res.resampling is True:
            rospy.loginfo(f"Resampling took place in step {step_res.step_idx}")

        if step_res.scan_match_failed is True:
            rospy.loginfo(f"Scan Mathing failed in step {step_res.step_idx}!")

        if step_res.scan_match_failed_fallback is True:
            rospy.loginfo(f"Scan match fallback failed in step {step_res.step_idx}!")
        


    def exe(self) -> None:
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

                step_time = msg.header.stamp.to_sec()

                # Run rbpf filter
                self._run_filter(
                    laser_scan=laser_scan,
                    dl=dl,
                    dr=dr,
                )
                
                step_duration = time.perf_counter() - start_time

                # Evaluate step results
                step_res, info = self._evaluate_run(
                    step_time=step_time,
                    step_duration=step_duration,
                    msg_queue_size=msg_queue_size
                )

                # Publish data
                self._publish_data(
                    step_res=step_res,
                    info=info
                )

                self.display_information(step_res=step_res)
                                
            except Exception as e:
                rospy.logerr(f"\nError processing RBPF input message:\n{e}")

            finally:
                self.rbpf_input_queue.task_done()
                
                        

def init():
    # Init ros node
    rospy.init_node(NODE_NAME)

    # Load ROS parameters
    robot_start_pose = load_ros_params()
    ros_params = ROSParams()
    
    # Define experiment parameters
    exp_params = def_exp_params(start_pose=robot_start_pose)

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


if __name__ == "__main__":
    main()
