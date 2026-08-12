#!/usr/bin/env python3
import debugpy

import rospy
import tf2_ros

from queue import Empty, Full, Queue

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, TransformStamped, Pose2D as Pose2DMsg
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from rbpf_slam.msg import RBPFInput


from typing import Tuple
from dataclasses import dataclass
import numpy as np

# Import classes (support both roslaunch and direct execution contexts)
try:
    from rbpf_slam.src.slam.scan_matcher.ogm_scan_matching import OGM
    from rbpf_slam.src.slam.infrastructure.defs import Pose2D
    from rbpf_slam.src.slam.rbpf.scan_match_factory import (
        OccupancyParams,
        SensorParams,
        MapParameter,
    )
except ModuleNotFoundError:
    from slam.scan_matcher.ogm_scan_matching import OGM
    from slam.infrastructure.defs import Pose2D
    from slam.rbpf.scan_match_factory import (
        OccupancyParams,
        SensorParams,
        MapParameter,
    )

'''
Description
-----------

This Script implements a ROS node for running the known-pose occupancy grid mapping algorithm (OGM). The class
OGMROSNode is a wrapper for the actual OGM algorithm, providing ROS-specific functionality such as subscribing
to input topics, publishing the occupancy grid map, and broadcasting transforms.

'''


# Constants
# Decide whether to use the debugger (True) or not (False).
USE_DEBUGGER = False
# Defines the name of the OGM Node
NODE_NAME = "ogm_node"



def debug_code():
    '''Starts debugpy and waits for debugger to attach.'''
    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()
    print("Debugger attached")


@dataclass
class Colormap:
    '''
    Dataclass that defines the discretized color values used to represent
    occupancy grid map cells after transformation.
    '''
    col_val_unknown: int = -1
    col_val_occ: int = 100
    col_val_free: int = 0


@dataclass
class ROSParams:
    '''
    Dataclass that defines the ROS parameters for the OGM node. These include
    topic names, TF frame names, and runtime settings.
    '''
    rbpf_input_topic: str
    map_topic: str
    map_tf_frame: str
    odom_tf_frame: str
    base_tf_frame: str
    laser_tf_frame: str
    input_queue_size: int


@dataclass
class OGMParams:
    '''
    Dataclass that defines the occupancy grid mapping experiment parameters,
    including map, sensor, occupancy, and discretization settings.
    '''
    occupancy_params: OccupancyParams
    sensor_params: SensorParams
    map_param: MapParameter
    occ_thresh: float
    free_thresh: float


def load_experiment_params() -> OGMParams:
    '''
    Loads the OGM experiment configuration from the ROS parameter server and
    initializes the OGMParams dataclass.

    Returns
    -------
    ogm_params : OGMParams
        The loaded occupancy grid mapping experiment parameters.

    Raises
    ------
    RuntimeError
        If the experiment configuration is missing required values or contains
        values that cannot be converted to the expected types.
    '''
    config = rospy.get_param("~experiment")

    try:
        discretization = config["discretization"]

        return OGMParams(
            occupancy_params=OccupancyParams(
                **config["occupancy_params"]
            ),
            sensor_params=SensorParams(
                **config["sensor_params"]
            ),
            map_param=MapParameter(
                **config["map_params"]
            ),
            occ_thresh=float(discretization["occ_thresh"]),
            free_thresh=float(discretization["free_thresh"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid OGM experiment configuration: {exc}"
        ) from exc


def load_colormap_params() -> Colormap:
    '''
    Loads the OccupancyGrid colormap values from the ROS parameter server and
    initializes the Colormap dataclass.

    Returns
    -------
    col_map : Colormap
        The color values used for unknown, occupied, and free map cells.

    Raises
    ------
    RuntimeError
        If the colormap configuration is missing required values or contains
        values that cannot be converted to integers.
    '''
    config = rospy.get_param("~colormap")

    try:
        return Colormap(
            col_val_unknown=int(config["col_val_unknown"]),
            col_val_occ=int(config["col_val_occ"]),
            col_val_free=int(config["col_val_free"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid OGM colormap configuration: {exc}"
        ) from exc


def load_ros_node_params() -> ROSParams:
    '''
    Loads the ROS topics, frames, and runtime settings from the ROS parameter
    server and initializes the ROSParams dataclass.

    Returns
    -------
    ros_params : ROSParams
        The loaded ROS parameters for the OGM node.

    Raises
    ------
    RuntimeError
        If the ROS configuration is missing required values or contains values
        that cannot be converted to the expected types.
    '''
    config = rospy.get_param("~ros")

    try:
        topics = config["topics"]
        frames = config["frames"]
        runtime = config["runtime"]

        return ROSParams(
            rbpf_input_topic=topics["rbpf_input"],
            map_topic=topics["map"],
            map_tf_frame=frames["map"],
            odom_tf_frame=frames["odom"],
            base_tf_frame=frames["base"],
            laser_tf_frame=frames["laser"],
            input_queue_size=int(runtime["input_queue_size"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid OGM ROS configuration: {exc}"
        ) from exc


def init_ogm(exp_param: OGMParams) -> OGM:
    '''
    Initializes an empty occupancy grid map from the given experiment
    parameters.

    Parameters
    ----------
    exp_param : OGMParams
        The occupancy, sensor, map, and discretization parameters of the OGM
        experiment.

    Returns
    -------
    ogm : OGM
        The initialized occupancy grid mapping algorithm.
    '''
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


class OGMROSNode:
    '''
    ROS interface for the known-pose occupancy grid mapping algorithm.

    Integrates the OGM algorithm into the robot's ROS infrastructure. The node
    receives synchronized laser scans and true robot poses, updates the map,
    publishes the discretized map as an OccupancyGrid message, and broadcasts
    the odom-to-base transform required to connect the robot's TF tree.

    Parameters
    ----------
    ogm : OGM
        The occupancy grid mapping algorithm used to update the map.
    ogm_params : OGMParams
        Configuration parameters of the occupancy grid mapping algorithm.
    ros_params : ROSParams
        ROS-specific configuration including topic names, frame names, and
        queue size.
    col_map : Colormap
        Color values used to discretize the log-odds map for publication.
    '''
    def __init__(
        self,
        ogm: OGM,
        ogm_params: OGMParams,
        ros_params: ROSParams,
        col_map: Colormap,
    ):
        # Set member variables
        self.ogm = ogm
        self.ogm_params = ogm_params
        self.ros_params = ros_params
        self.col_map = col_map

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

        # Define publisher for the discretized occupancy grid map.
        self.map_publisher = rospy.Publisher(
            name=self.ros_params.map_topic,
            data_class=OccupancyGrid,
            queue_size=1,
            latch=True,
        )

        # Colorization
        self.col_map_poitns = rospy.Publisher("debug_cells", Marker, queue_size=1)


    def lookup_base_to_laser_transform_2d(self):
        '''
        Looks up the static transform from the source laser frame into the
        target base frame and converts it into a planar pose.

        The lookup is repeated until the transform becomes available or ROS is
        shut down.

        Returns
        -------
        Pose2D
            The laser pose in the base frame as an (x, y, yaw) tuple. Returns a
            zero pose if ROS shuts down before the transform is available.
        '''
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
        '''
        Stores a synchronized RBPF input message in the processing queue while
        preserving the message order.

        Parameters
        ----------
        msg : RBPFInput
            The synchronized laser scan, wheel encoder, and true-pose data
            produced by the RBPF data processor node.
        '''
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
        orient_yaw: float = 0.0,
    ) -> OccupancyGrid:
        '''
        Creates a ROS OccupancyGrid message from a flattened map and its
        metadata.

        Parameters
        ----------
        map_raveled : np.ndarray
            The flattened occupancy grid map in row-major order.
        timestamp : rospy.Time
            The timestamp for the OccupancyGrid message. If None, the current
            ROS time is used.
        frame_id : str
            The frame ID in which the map is represented.
        grid_res : float
            The grid resolution in meters per cell.
        width : int
            The map width in number of cells.
        height : int
            The map height in number of cells.
        origin_x : float
            The x-coordinate of the map origin in the map frame.
        origin_y : float
            The y-coordinate of the map origin in the map frame.
        orient_yaw : float, optional
            The yaw orientation of the map origin in radians. Default is 0.0.

        Returns
        -------
        OccupancyGrid
            A ROS OccupancyGrid message containing the map data and metadata.
        '''
        map_msg = OccupancyGrid()

        timestamp = timestamp if timestamp is not None else rospy.Time.now()

        map_msg.header.stamp = timestamp
        map_msg.header.frame_id = frame_id
        map_msg.data = map_raveled

        map_msg.info.map_load_time = timestamp
        map_msg.info.resolution = grid_res
        map_msg.info.width = width
        map_msg.info.height = height

        map_msg.info.origin.position.x = origin_x
        map_msg.info.origin.position.y = origin_y
        map_msg.info.origin.position.z = 0.0

        quat = quaternion_from_euler(0.0, 0.0, orient_yaw)
        map_msg.info.origin.orientation.x = quat[0]
        map_msg.info.origin.orientation.y = quat[1]
        map_msg.info.origin.orientation.z = quat[2]
        map_msg.info.origin.orientation.w = quat[3]

        return map_msg


    @staticmethod
    def transform_pose_to_planar_pose(pose: Pose2DMsg):
        '''
        Converts a ROS Pose2D message into a planar pose tuple.

        Parameters
        ----------
        pose : Pose2DMsg
            The ROS pose message to convert.

        Returns
        -------
        Pose2D
            The pose represented as an (x, y, theta) tuple.
        '''
        return (pose.x, pose.y, pose.theta)


    @staticmethod
    def transform_laser_scan_to_measurement(laser_scan: LaserScan):
        '''
        Converts a ROS LaserScan message into range-bearing measurements.

        Parameters
        ----------
        laser_scan : LaserScan
            The laser scan message to convert.

        Returns
        -------
        measurements : list
            The scan represented as a list of (range, bearing) tuples.
        '''
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
        '''
        Composes a planar robot pose with a planar sensor offset.

        Parameters
        ----------
        pose : Pose2D
            The base pose in the world frame as an (x, y, yaw) tuple.
        pose_offset : Pose2D
            The laser pose relative to the base frame as an (x, y, yaw) tuple.

        Returns
        -------
        Pose2D
            The laser pose in the world frame as an (x, y, yaw) tuple.
        '''
        x, y, yaw = pose
        dx, dy, dyaw = pose_offset
        transformed_x = x + np.cos(yaw) * dx - np.sin(yaw) * dy
        transformed_y = y + np.sin(yaw) * dx + np.cos(yaw) * dy
        transformed_yaw = yaw + dyaw
        return (transformed_x, transformed_y, transformed_yaw)


    def convert_log_odds_map(self, log_odds_map: np.ndarray) -> np.ndarray:
        '''
        Converts a log-odds map into a discretized color-value map suitable for
        publication as a ROS OccupancyGrid message.

        Parameters
        ----------
        log_odds_map : np.ndarray
            The log-odds map represented as a 2D numpy array.

        Returns
        -------
        discretized_map : np.ndarray
            A 2D array whose cells contain the configured unknown, free, or
            occupied values. Returns None when no log-odds map is available.
        '''
        if log_odds_map is None:
            rospy.logwarn("No log-odds map available. Skipping map publishing.")
            return None

        return OGM.discretize_map(
            ogm=log_odds_map,
            occ_thres=self.ogm_params.occ_thresh,
            free_thres=self.ogm_params.free_thresh,
            col_val_unknown=self.col_map.col_val_unknown,
            col_val_free=self.col_map.col_val_free,
            col_val_occ=self.col_map.col_val_occ,
        )


    def publish_occupancy_grid_message(self, timestamp: rospy.Time) -> None:
        '''
        Discretizes the current OGM map, creates a ROS OccupancyGrid message,
        and publishes it on the configured map topic.

        Parameters
        ----------
        timestamp : rospy.Time
            The timestamp assigned to the published map and its metadata.
        '''
        log_odds_map = self.ogm.get_log_odds_map()
        map_meta = self.ogm.get_map_meta()

        discretized_map = self.convert_log_odds_map(log_odds_map)
        if discretized_map is None:
            return

        map_msg = self._map_into_occupancy_grid_msg(
            map_raveled=discretized_map.ravel(order="C").tolist(),
            timestamp=timestamp,
            frame_id=self.ros_params.map_tf_frame,
            grid_res=float(map_meta.get("grid_resolution_m")),
            width=int(map_meta.get("number_of_cells_x")),
            height=int(map_meta.get("number_of_cells_y")),
            origin_x=float(-map_meta.get("shift_x")),
            origin_y=float(-map_meta.get("shift_y")),
            orient_yaw=0.0,
        )

        self.map_publisher.publish(map_msg)



    def publish_green_cells(self, i_range, j_range, ogm):        
        '''
        Publishes the selected occupancy grid cells as green visualization
        markers for debugging.

        Parameters
        ----------
        i_range : Tuple[int, int]
            The half-open range of map row indices to visualize.
        j_range : Tuple[int, int]
            The half-open range of map column indices to visualize.
        ogm : OGM
            The occupancy grid map used to convert cell indices into map-frame
            coordinates and determine marker size.
        '''
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

        
    def exe(self):
        '''
        Main loop that executes the OGM node.

        Retrieves synchronized input messages from the queue, transforms the
        robot pose into the laser pose, updates and extends the occupancy grid
        map, publishes the odom-to-base transform, and publishes the current
        discretized map.
        '''
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
                self.publish_occupancy_grid_message(timestamp=msg.header.stamp)

            finally:
                self.rbpf_input_queue.task_done()



def init():
    '''
    Initializes the OGM algorithm and its configuration. The following steps
    are performed:
        1. Initialize the ROS node.
        2. Load the experiment, ROS, and colormap parameters.
        3. Initialize the occupancy grid mapping algorithm.
        4. Return the initialized components.

    Returns
    -------
    ogm : OGM
        The initialized occupancy grid mapping algorithm.
    exp_params : OGMParams
        The loaded OGM experiment parameters.
    ros_params : ROSParams
        The loaded ROS node parameters.
    col_map : Colormap
        The loaded color values used to discretize the map.
    '''
    # Init ROS node
    rospy.init_node(NODE_NAME)

    # Load configuration
    config_name = rospy.get_param("~config_name")
    exp_params = load_experiment_params()
    ros_params = load_ros_node_params()
    col_map = load_colormap_params()

    # Init OGM
    ogm = init_ogm(exp_param=exp_params)

    rospy.loginfo("Loaded OGM configuration: %s", config_name)

    return ogm, exp_params, ros_params, col_map


def main():
    '''Initializes the OGM ROS node and starts its processing loop.'''
    # Debug code if enabled
    if USE_DEBUGGER:
            debug_code()

    # Initialize node and parameters
    ogm, exp_params, ros_params, col_map = init()

    # Initialize algorithm
    ros_ogm = OGMROSNode(
        ogm=ogm,
        ogm_params=exp_params,
        ros_params=ros_params,
        col_map=col_map,
    )

    # Start the algorithm
    ros_ogm.exe()

    

if __name__=="__main__":
    main()
