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
        Returns a log odds map mgit essage object containing the map and the map metadata.
        '''
        return self.ogm.return_log_odds_map_object()


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
        transf_param = self.icp.find_transformation(
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

        Parameters:
        ---------
        old_pose: Pose2D
            The previous pose of the robot.
        dl: float
            The distance traveled by the left wheel since the last update.
        dr: float
            The distance traveled by the right wheel since the last update.
        measurements: List[Tuple[float, float]]
            A list of tuples containing the range and bearing measurements from the robot's sensors.
        
        Returns:
        ---------
        Tuple[Pose2D, Pose2D]
            A tuple containing the corrected pose and the predicted pose, in that order. If scan matching cannot be
            performed by any means, the predicted pose is returned for both values.   
        ''' 
        profile_totals_ns: dict[str, int] = {}
        profile_counts: dict[str, int] = {}

        def _profile_accumulate(metric_name: str, start_ns: int) -> int:
            elapsed_ns = time.perf_counter_ns() - start_ns
            profile_totals_ns[metric_name] = profile_totals_ns.get(metric_name, 0) + elapsed_ns
            profile_counts[metric_name] = profile_counts.get(metric_name, 0) + 1
            return elapsed_ns

        def _profile_print_summary() -> None:
            if not profile_totals_ns:
                return

            avg_profile_ns = [
                (name, profile_totals_ns[name] / max(profile_counts.get(name, 1), 1), profile_counts.get(name, 0), profile_totals_ns[name])
                for name in profile_totals_ns
            ]
            avg_profile_ns.sort(key=lambda x: x[1], reverse=True)

            print("\n[ScanMatcher Profiling] update_pose avg time per measured block (ms), sorted high -> low")
            for name, avg_ns, n_calls, total_ns in avg_profile_ns:
                print(
                    f"  {name}: avg={avg_ns / 1e6:.6f} ms | calls={n_calls} | total={total_ns / 1e6:.6f} ms"
                )

        t_update_pose_total_ns = time.perf_counter_ns()

        # Init pose
        pred_pose = None
        corr_pose = None

        # Predict psoe based on wheel encoder information
        t_ns = time.perf_counter_ns()
        pred_pose = self.predict_pose(
            pose=old_pose,
            dl=dl,
            dr=dr,
        )
        _profile_accumulate("predict_pose_ns", t_ns)

        if len(measurements) < 3:
            self.pose = pred_pose
            _profile_accumulate("update_pose_total_ns", t_update_pose_total_ns)
            _profile_print_summary()
            return corr_pose, pred_pose
        
        # Find max measurement range
        max_meas_range = max([m[0] for m in measurements])

        # Transform measurements (range, bearing) -> point cloud
        t_ns = time.perf_counter_ns()
        scan_points = self.transform_measurements_to_points(
            pose=pred_pose,
            measurements=measurements
        )
        _profile_accumulate("transform_measurements_to_points_ns", t_ns)

        scan_points = scan_points[np.all(np.isfinite(scan_points), axis=1)]

        if scan_points.shape[0] < 3:
            self.pose = pred_pose
            _profile_accumulate("update_pose_total_ns", t_update_pose_total_ns)
            _profile_print_summary()
            return corr_pose, pred_pose

        # Get map points
        t_ns = time.perf_counter_ns()
        # map_points = self.ogm.extract_map_for_scan_matching(
        #     pose=pred_pose,
        #     radius=max_meas_range,
        #     delta_r=self.delta_r,
        #     occ_thresh=self.occ_thres,
        # )
        
        # map_points = self.ogm.extract_map_for_scan_matching_np_optm(
        #     pose=pred_pose,
        #     radius=max_meas_range,
        #     delta_r=self.delta_r,
        #     occ_thresh=self.occ_thres,
        # )

        map_points = self.ogm.extract_map_for_scan_matching_numba(
            pose=pred_pose,
            radius=max_meas_range,
            delta_r=self.delta_r,
            occ_thresh=self.occ_thres,
        )

        _profile_accumulate("extract_map_for_scan_matching_ns", t_ns)

        # Filter map points -> only finite and shape must be valid
        map_points = np.asarray(map_points, dtype=float)
        map_points = map_points[np.all(np.isfinite(map_points), axis=1)]
        if map_points.ndim != 2 or map_points.shape[0] < 3:
            self.pose = pred_pose
            _profile_accumulate("update_pose_total_ns", t_update_pose_total_ns)
            _profile_print_summary()
            return corr_pose, pred_pose


        # Correct pose
        t_ns = time.perf_counter_ns()
        corr_pose = self.correct_pose(
            pose=pred_pose, 
            scan_points=scan_points,
            map_points=map_points,
        )
        _profile_accumulate("correct_pose_ns", t_ns)

        self.pose = corr_pose

        _profile_accumulate("update_pose_total_ns", t_update_pose_total_ns)
        _profile_print_summary()

        return corr_pose, pred_pose
    


    def update_pose_copy(
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

        Parameters:
        ---------
        old_pose: Pose2D
            The previous pose of the robot.
        dl: float
            The distance traveled by the left wheel since the last update.
        dr: float
            The distance traveled by the right wheel since the last update.
        measurements: List[Tuple[float, float]]
            A list of tuples containing the range and bearing measurements from the robot's sensors.
        
        Returns:
        ---------
        Tuple[Pose2D, Pose2D]
            A tuple containing the corrected pose and the predicted pose, in that order. If scan matching cannot be
            performed by any means, the predicted pose is returned for both values.   
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
        
        # Find max measurement range
        max_meas_range = max([m[0] for m in measurements])

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
            radius=max_meas_range,
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