#!/usr/bin/env python3

import os
import sys
import time

# For math
import numpy as np
from math import sin, cos, pi, atan2

from typing import Tuple, List

Pose2D = Tuple[float, float, float]

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# if SCRIPT_DIR not in sys.path:
#     sys.path.insert(0, SCRIPT_DIR)

# Import ICP
from .icp_scan_matching import IterativeClosestPoint

# Import OGM
from .ogm_scan_matching import OGM



class ScanMatcher():
    def __init__(
            self,
            ogm: OGM,
            icp: IterativeClosestPoint, 
        robo_param: float,
        sensor_parameters: Tuple[float, float],
        occ_thres: float,
        surface_radius_m: float = 0.1,
        min_free_ratio: float = 0.25,
    ):
        # Extract parameter
        self.ogm = ogm
        self.icp = icp
        initial_robo_pose, wheel_separation = robo_param
        min_sensor_range, max_sensor_range, delta_r = sensor_parameters

        # Init members
        self.pose = initial_robo_pose
        self.wheel_separation = wheel_separation
        self.min_sensor_range = min_sensor_range
        self.max_sensor_range = max_sensor_range
        self.delta_r = delta_r
        self.occ_thres = occ_thres
        self.surface_radius_m = surface_radius_m
        self.min_free_ratio = min_free_ratio
        self.last_pred_pose = None
        self.last_map_points_count = 0
        self.last_t_scan_matching_s = None
        self.last_t_prediction_s = None
        self.last_t_map_extraction_s = None
        self.last_t_correct_pose_s = None


    def get_pose(self) -> Pose2D:
        '''
        Returns the pose of robot. Attention, not guaranteed the corrected one. Best to call directly
        after 'update_pose()' call.
        '''
        return self.pose


    def get_ogm(self):
        '''
        Returns a log odds map message object containing the map and the map metadata.
        '''
        return self.ogm.return_log_odds_map_object()


    def get_trained_nn_tree(self):
        '''
        Returns the NN tree that has been trained on the map points, if exists, otherwise returns None. 
        '''
        if hasattr(self.icp.neighbor, "_fit_X"):
            return self.icp.neighbor
        else:
            return None



    def get_info(self) -> dict:
        '''
        Returns a dictionary containing the current state of the scan matcher. This includes the current pose, 
        the current map and the current state of the ICP stop condition.

        Returns:
            dict: A dictionary containing the current state of the scan matcher.
                {
                    "iteration": int,
                    "mean_err": float,
                    "min_mean_err": float,
                    "rel_improvement": float,
                    "no_improvement_counter": int,
                    "dtrans_norm": float,
                    "drot_abs": float,
                    "stop_reason": str,
                    "max_correspondence_distance": float,
                    "min_squared_error": float,                
                    "n_points_true_data": int,
                    "n_points_new_data": int,
                    "transformed_new_data_list": List[np.ndarray],  # List of transformed new data at each iteration
                    "squared_error_list": List[float],  # List of squared errors at each iteration
                    "transformation_parameter_list": List[np.ndarray],  # List of transformation parameters at each iteration
                    "list_of_cleaned_corresp": List[List[Tuple[int, int]]],  # List of cleaned correspondences at each iteration
                    "list_of_cleaned_corresp_numb": List[int],  # List of number of cleaned correspondences at each iteration
                    "list_of_corresp_numb": List[int],  # List of number of correspondences at each iteration
                    "scan_match_pose": Pose2D,
                }

        '''
        info = self.icp.get_info()

        info["scan_match_pose"] = self.pose
        info["pred_pose"] = self.last_pred_pose
        info["map_points_count"] = int(self.last_map_points_count)
        info["time_duration_scan_matching"] = self.last_t_scan_matching_s
        info["time_duration_prediction"] = self.last_t_prediction_s
        info["time_duration_map_extraction"] = self.last_t_map_extraction_s
        # TODO: Replace this with useful thing. Dont use timing to detect this
        info["time_duration_correct_pose"] = self.last_t_correct_pose_s
        return info
    

    def transform_measurements_to_points(
            self, 
            pose: Pose2D,
            measurements: List[Tuple[float, float]],
    ) -> np.ndarray:
        '''
        Get's a 2d pose and a list of measurements (range, bearing) and transforms them to a point cloud in 
        map frame, based on the given pose.

        Parameters
        ----------
        pose: Pose2D
            The pose of the robot in the map frame, given as a tuple (x, y, theta).
        measurements: List[Tuple[float, float]]
            A list of tuples containing the range and bearing measurements from the robot's sensors

        Returns
        -------
        np.ndarray
            A 2D numpy array of shape (N, 2) containing the x and y coordinates of the points in the map frame, 
            where N is the number of measurements.
        '''
        # Extract pose
        x, y, theta = pose

        # Extract measurement to vectors
        measurements = np.array(measurements)
        r = measurements[:, 0]
        b = measurements[:,1]

        # Compute phi
        phi = theta + b

        # Compute points from scan
        c = np.cos(phi)
        s = np.sin(phi)
        x_points = x + r * c
        y_points = y + r * s

        return np.column_stack((x_points, y_points))


    def predict_pose(self, pose: Pose2D, dl: float, dr: float) -> Pose2D:
        '''
        Predicts the pose based in the given control values and the wheel separation of the DDMR.

        Parameters
        ----------
        pose: Pose2D
            The current pose of the robot, given as a tuple (x, y, theta).
        dl: float
            The distance traveled by the left wheel since the last update.
        dr: float
            The distance traveled by the right wheel since the last update. 
        
        Returns
        -------
        Pose2D
            The predicted pose of the robot (x, y, theta) 

        '''
        # Extract pose
        x, y, theta = pose
        
        # predict pose for the case that we turned
        if dr != dl:
            alpha = (dr - dl) / self.wheel_separation
            rad = dl/alpha
            g1 = x + (rad + self.wheel_separation/2.)*(sin(theta+alpha) - sin(theta))
            g2 = y + (rad + self.wheel_separation/2.)*(-cos(theta+alpha) + cos(theta))
            g3 = (theta + alpha + pi) % (2*pi) - pi

        # Predict pose for the case we drove on a straight line
        else:
            g1 = x + dl * cos(theta)
            g2 = y + dl * sin(theta)
            g3 = theta
        
        return (g1, g2, g3)


    def correct_pose(self, pose: Pose2D, scan_points: np.ndarray, map_points: np.ndarray) -> Pose2D:
        '''
        Corrects the robots pose by scan matching the measurement against the current map. 

        Parameters
        ----------
        pose: Pose2D
            The current pose of the robot, given as a tuple (x, y, theta).
        scan_points: np.ndarray
            A 2D numpy array of shape (N, 2) containing the x and y coordinates of the points in the scan, 
            where N is the number of points.
        map_points: np.ndarray
            A 2D numpy array of shape (N, 2) containing the x and y coordinates of the points in the map, 
            where N is the number of points.

        Returns
        -------
        Pose2D
            The corrected pose of the robot (x, y, theta) 

        '''
        # Find best transformation for given points
        result = self.icp.find_transformation(
            new_data_pointpairs=scan_points,
            true_data_pointpairs=map_points,
        )

        if not result.use_transformation:
            return None 

        # Transform pose -> Correction
        pose = self.icp.correct_pose(
            pose=pose,
            transf_param=result.transformation,
        )

        return pose


    def update_pose(
            self,
            old_pose: Pose2D,
            dl: float, dr: float, 
            measurements: List[Tuple[float, float]],
    ) -> Tuple[Pose2D, Pose2D]:
        '''
        Updates the pose of the robot by first predicting it based on the control values and then correcting it
        by scan matching the measurement against the current map.

        Returns the corrected pose first and the predicted pose second. If scan matching cannot be
        performed safely, the corrected pose is None. In this case the self.pose member will be set to the predicted pose.
        If succeed the self.member will be set to the corrected pose.

        Parameters
        ---------
        old_pose: Pose2D
            The previous pose of the robot.
        dl: float
            The distance traveled by the left wheel since the last update.
        dr: float
            The distance traveled by the right wheel since the last update.
        measurements: List[Tuple[float, float]]
            A list of tuples containing the range and bearing measurements from the robot's sensors.
        
        Returns
        ---------
        Tuple[Pose2D, Pose2D]
            A tuple containing the corrected pose and the predicted pose, in that order. If scan matching cannot be
            performed by any means, the corrected pose is None and the predicted pose is returned as the second element
            of the tuple.    
        ''' 
        t_scan_matching_start = time.perf_counter()

        def _finish_and_return(corr_pose_local, pred_pose_local):
            self.last_t_scan_matching_s = time.perf_counter() - t_scan_matching_start
            return corr_pose_local, pred_pose_local

        # Init pose
        pred_pose = None
        corr_pose = None
        self.last_map_points_count = 0
        self.last_t_prediction_s = None
        self.last_t_map_extraction_s = None
        self.last_t_correct_pose_s = None

        # Predict psoe based on wheel encoder information
        t_prediction_start = time.perf_counter()
        pred_pose = self.predict_pose(
            pose=old_pose,
            dl=dl,
            dr=dr,
        )
        self.last_t_prediction_s = time.perf_counter() - t_prediction_start
        self.last_pred_pose = pred_pose

        if len(measurements) < 3:
            self.pose = pred_pose
            return _finish_and_return(corr_pose, pred_pose)
        
        # Find max measurement range
        max_meas_range = max([m[0] for m in measurements])

        t_map_extraction_start = time.perf_counter()

        # Transform measurements (range, bearing) -> point cloud
        scan_points = self.transform_measurements_to_points(
            pose=pred_pose,
            measurements=measurements
        )

        # Filter inf and nan values from measurements and check if enough scans are left, else break
        scan_points = scan_points[np.all(np.isfinite(scan_points), axis=1)]
        self.last_t_map_extraction_s = time.perf_counter() - t_map_extraction_start
        if scan_points.shape[0] < 3:
            self.pose = pred_pose
            return _finish_and_return(corr_pose, pred_pose)

        # Get map points
        map_points = self.ogm.extract_map_for_scan_matching_numba(
            pose=pred_pose,
            radius=max_meas_range,
            delta_radius=self.delta_r,
            occ_thresh=self.occ_thres,
            surface_radius_m=self.surface_radius_m,
            min_free_ratio=self.min_free_ratio,
        )
        self.last_map_points_count = int(map_points.shape[0]) if map_points.ndim == 2 else 0

        # Check if array shape is correct and has enough elements, else break
        if map_points.ndim != 2 or map_points.shape[0] < 3:
            self.pose = pred_pose
            return _finish_and_return(corr_pose, pred_pose)

        # Correct pose
        t_correct_pose_start = time.perf_counter()
        corr_pose = self.correct_pose(
            pose=pred_pose, 
            scan_points=scan_points,
            map_points=map_points,
        )
        self.last_t_correct_pose_s = time.perf_counter() - t_correct_pose_start

        # Keep a valid pose even when ICP correction is rejected.
        self.pose = corr_pose if corr_pose is not None else pred_pose

        return _finish_and_return(corr_pose, pred_pose)



def main():
    pass


if __name__ == "__main__":
    main()