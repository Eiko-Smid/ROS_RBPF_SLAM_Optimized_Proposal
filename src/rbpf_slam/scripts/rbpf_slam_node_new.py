#!/usr/bin/env python3
import debugpy

from typing import List, Tuple

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

    from slam.optimize_rbpf_multiple_particles.playback_defs import ExperimentParams, PlaybackData
    from slam.infrastructure.playback_recorder import PlaybackRecorder, build_metadata


NODE_NAME = "rbpf_slam_node"

'''

TODO: Load start pose

TODO: Read wheel separation from robot description or ROS parameter server.


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



class RBPF_ROS_Node:
    def __init__(self, rbpf: RBPF, exp_params: ExperimentParams, ros_params: ROSParams):
        self.rbpf = rbpf
        self.exp_params = exp_params
        self.ros_params = ros_params

        # Define obj to locks threads
        self.lock = threading.Lock()

        # Initialize ROS publishers and subscribers
        # Init subscirber and Queue to store input data
        self.rbpf_input_queue = Queue(maxsize=self.ros_params.input_queue_size)
        self.rbpf_input_sub = rospy.Subscriber(
            name=self.ros_params.rbpf_input_topic,
            data_class=RBPFInput,
            callback=self._rbpf_input_callback,
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


    def _rbpf_input_callback(self, msg: RBPFInput) -> None:
        # Store message into queue for processing in the main loop       
        try:
            self.rbpf_input_queue.put_nowait(msg)
        except Full:
            rospy.logerr_throttle(
                period=2.0,
                msg="RBPF input queue is full. The filter cannot keep up.",
            )


    def exe(self) -> None:
        while not rospy.is_shutdown():
            # Extract ne data from queue
            try:
                msg: RBPFInput = self.rbpf_input_queue.get(timeout=0.1)
            except Empty:    
                continue

            try:
                # Extract data from message
                laser_scan = msg.laser_scan
                dl = msg.wheel_encoder.left
                dr = msg.wheel_encoder.right

                # Transform measurements to (range, bearing) tuples
                measurements = self.transform_laser_scan_to_measurement(laser_scan)

                # Subsample and clean measuremnts for filter and map
                every_nth_scan_filter = self.exp_params.every_nth_scan_filter
                measurements_filter = (
                    measurements[::every_nth_scan_filter] if every_nth_scan_filter > 1 else measurements
                )

                measurements_filter = [
                    (r, b) for r, b in measurements_filter if np.isfinite(r)
                ]

                every_nth_scan_map = self.exp_params.every_nth_scan_map
                measurements_map = (
                    measurements[::every_nth_scan_map] if every_nth_scan_map > 1 else measurements
                )

                # Do rbpf update step
                self.rbpf.step_range_finder_model(
                    odom=(dl, dr),
                    measurements_proposal=measurements_filter,
                    measurements_map=measurements_map,
                    proposal_sigma_xy=self.exp_params.proposal_sigma_xy,
                    proposal_sigma_theta=self.exp_params.proposal_sigma_theta,
                    cov_std_scale=self.exp_params.cov_std_scale,
                    cov_max_std_xy=self.exp_params.cov_max_std_xy,
                    cov_max_std_theta=self.exp_params.cov_max_std_theta,
                    min_std_xy=self.exp_params.min_std_xy,
                    min_std_theta=self.exp_params.min_std_theta,
                )
                
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
    
    exp_params = def_exp_params(start_pose=robot_start_pose)

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

    return rbpf, exp_params, ros_params
    


def main():
    if USE_DEBUGGER:
        debug_code()

    # Init RBPF filter
    rbpf, exp_params, ros_params = init()


if __name__ == "__main__":
    main()
