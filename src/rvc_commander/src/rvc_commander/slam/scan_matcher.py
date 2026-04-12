#!/usr/bin/env python3

import os
import sys

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
            robo_param: float, sensor_parameters: Tuple[float, float], occ_thres: float
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


    def get_info(self) -> dict:
        '''
        Returns a dictionary containing the current state of the scan matcher. This includes the current pose, 
        the current map and the current state of the ICP stop condition.
        '''
        info = self.icp.get_info()

        info["scan_match_pose"] = self.pose
        return info


    def transform_measurements_to_points(
            self, 
                pose: Pose2D,
            measurements: List[Tuple[float, float]],
    ) -> np.ndarray:
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
        '''
        # Find best transformation for given points
        transf_param, _, _, _ = self.icp.find_transformation(
            new_data_pointpairs=scan_points,
            true_data_pointpairs=map_points,
        )

        # Transform pose -> Correction
        pose = self.icp.correct_pose(
            pose=pose,
            transf_param=transf_param,
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
        performed safely, the predicted pose is returned for both values.
        ''' 
        # Init pose
        pred_pose = None
        corr_pose = None

        # Predict psoe based on wheel encoder information
        pred_pose = self.predict_pose(
            pose=old_pose,
            dl=dl,
            dr=dr,
        )

        if len(measurements) < 3:
            self.pose = pred_pose
            return corr_pose, pred_pose

        # Transform measurements (range, bearing) -> point cloud
        scan_points = self.transform_measurements_to_points(
            pose=pred_pose,
            measurements=measurements
        )

        scan_points = scan_points[np.all(np.isfinite(scan_points), axis=1)]

        if scan_points.shape[0] < 3:
            self.pose = pred_pose
            return corr_pose, pred_pose

        # Get map points
        map_points = self.ogm.extract_map_for_scan_matching(
            pose=pred_pose,
            radius=self.max_sensor_range,
            delta_r=self.delta_r,
            occ_thresh=self.occ_thres,
        )

        # Filter map points -> only finite and shape must be valid
        map_points = np.asarray(map_points, dtype=float)
        map_points = map_points[np.all(np.isfinite(map_points), axis=1)]
        if map_points.ndim != 2 or map_points.shape[0] < 3:
            self.pose = pred_pose
            return corr_pose, pred_pose


        # Correct pose
        corr_pose = self.correct_pose(
            pose=pred_pose, 
            scan_points=scan_points,
            map_points=map_points,
        )

        self.pose = corr_pose

        return corr_pose, pred_pose



def main():
    pass


if __name__ == "__main__":
    main()