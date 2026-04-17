#!/usr/bin/env python3

from __future__ import annotations

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, atan2, pi, inf
from sklearn.neighbors import NearestNeighbors
from heapq import heappush, heappop



class ICPStopCondition:
    '''
    Class that checks if ICP should stop based on multiple criteria:
    - Max iterations
    - Relative improvement
    - No improvement limit
    - Absolute error threshold
    - Transformation magnitude threshold
    '''
    def __init__(
        self,
        max_iterations: int = 10,       # max number of iterations for icp
        epsilon_rel: float = 1e-3,      # minimum relative improvement to continue icp
        no_improvement_limit: int = 2,  # maximum number of iterations without improvement
        min_error: float = 1.0,         # minimum mean error threshold to continue icp
        min_dtrans: float = 1e-4,       # minimum translation change threshold to continue icp
        min_drot: float = 1e-1          # minimum rotation change threshold to continue icp (in radians)

    ):
        # Init params
        self.max_iterations = max_iterations
        self.epsilon_rel = epsilon_rel
        self.no_improvement_limit = no_improvement_limit
        self.min_error = min_error
        self.min_dtrans = min_dtrans
        self.min_drot = min_drot

        # internal state
        self.info: dict = {}
        self.prev_error = inf
        self.mean_err = inf
        self.no_improvement_counter = 0
        self.iteration = 0
        self.dtrans_norm = None
        self.drot_abs = None
        self.stop_reason = None
        self.rel_improvement = None
        self.no_improvement_counter = None  


    def get_stop_reason(self) -> str:
        '''
        Returns the reason why ICP stopped.'''
        return self.stop_reason


    def store_info(self):
        '''
        Stores the current state of the ICP stop condition and other relevant information in the 'info' member variable.
        '''
        self.info["iteration"] = self.iteration
        self.info["mean_err"] = self.mean_err
        self.info["rel_improvement"] = self.rel_improvement
        self.info["no_improvement_counter"] = self.no_improvement_counter
        self.info["min_mean_err"] = self.min_error
        self.info["dtrans_norm"] = self.dtrans_norm
        self.info["drot_abs"] = self.drot_abs
        self.info["stop_reason"] = self.stop_reason


    def get_info(self) -> dict:
        '''
        Returns a dictionary with the current state of the stop condition. 

        Returns:
        ----------
            dict: With the following structure:
            {
                "iteration": int,
                "mean_err": float,
                "rel_improvement": float,
                "no_improvement_counter": int,
                "dtrans_norm": float,
                "drot_abs": float,
                "stop_reason": str
            }
        '''
        return self.info


    def reset(self):
        '''
        Resets the stop condition state for a new ICP run.
        '''
        self.prev_error = inf
        self.mean_err = inf
        self.no_improvement_counter = 0
        self.iteration = 0
        self.dtrans_norm = None
        self.drot_abs = None
        self.stop_reason = None
        self.rel_improvement = None
        self.no_improvement_counter = None  


    def stop_icp(self, mean_err: float, dtransformation: np.ndarray) -> bool:
        """
        Checks if ICP should stop based on the current error and transformation change.

        Parameters:
        ----------
            mean_err: The current mean error metric (e.g., mean distance between correspondences).
            dtransformation: The change in transformation (translation and rotation) from the last iteration.
        
        Returns:
        ----------
            bool: True if ICP should stop, False otherwise.
        """
        no_improvement = False 
        self.mean_err = mean_err
        stop_icp_ = False

        if self.iteration > 0:
            # Stop cause: Max iterations
            if self.iteration >= self.max_iterations:
                self.stop_reason = "Max iterations reached"
                stop_icp_ = True 

            # Stop cause: Absolute error threshold
            if self.mean_err < self.min_error:
                self.stop_reason = "Absolute error threshold reached"
                stop_icp_ = True   

            # Check transformation magnitude
            if dtransformation is not None:
                
                if not np.all(np.isfinite(dtransformation)):
                    self.stop_reason = "Non-finite transformation detected"
                    stop_icp_ = True   
                
                # Compute translation magnitude
                self.dtrans_norm = np.linalg.norm(dtransformation[:2])
                self.drot_abs = abs(dtransformation[2])[0]
                
                if self.dtrans_norm < self.min_dtrans and self.drot_abs < self.min_drot:
                    self.stop_reason = "Transformation magnitude below threshold"
                    stop_icp_ = True   

            # Compute Relative improvement
            if not np.isfinite(self.prev_error):
                self.rel_improvement = float("inf")
            else:
                self.rel_improvement = abs(self.prev_error - self.mean_err) / max(self.prev_error, 1e-12)

            # Track improvement
            if self.rel_improvement < self.epsilon_rel:
                no_improvement = True

            # Divergence check 
            if self.mean_err > self.prev_error:
                no_improvement = True

            # Update improvement counter
            if no_improvement:
                self.no_improvement_counter += 1
            else:
                self.no_improvement_counter = 0

            # Check if improvement 
            if self.no_improvement_counter >= self.no_improvement_limit:
                self.stop_reason = "No improvement limit reached"
                stop_icp_ = True   
        
        # update state
        self.prev_error = self.mean_err

        if not stop_icp_:
            self.iteration += 1

        return stop_icp_
    

def stop_icp_copy(self, mean_err: float, dtransformation: np.ndarray) -> bool:
        """
        Checks if ICP should stop based on the current error and transformation change.

        Parameters:
        ----------
            mean_err: The current mean error metric (e.g., mean distance between correspondences).
            dtransformation: The change in transformation (translation and rotation) from the last iteration.
        
        Returns:
        ----------
            bool: True if ICP should stop, False otherwise.
        """
        no_improvement = False 
        self.mean_err = mean_err

        # Stop cause: Max iterations
        if self.iteration >= self.max_iterations:
            self.stop_reason = "Max iterations reached"
            return True

        # Stop cause: Absolute error threshold
        if self.mean_err < self.min_error:
            self.stop_reason = "Absolute error threshold reached"
            return True        

        if self.iteration > 0:
            # Check transformation magnitude
            if dtransformation is not None:
                
                if not np.all(np.isfinite(dtransformation)):
                    self.stop_reason = "Non-finite transformation detected"
                    return True
                
                # Compute translation magnitude
                self.dtrans_norm = np.linalg.norm(dtransformation[:2])
                self.drot_abs = abs(dtransformation[2])[0]
                
                if self.dtrans_norm < self.min_dtrans and self.drot_abs < self.min_drot:
                    self.stop_reason = "Transformation magnitude below threshold"
                    return True

            # Compute Relative improvement
            if not np.isfinite(self.prev_error):
                self.rel_improvement = float("inf")
            else:
                self.rel_improvement = abs(self.prev_error - self.mean_err) / max(self.prev_error, 1e-12)

            # Track improvement
            if self.rel_improvement < self.epsilon_rel:
                no_improvement = True

            # Divergence check 
            if self.mean_err > self.prev_error:
                no_improvement = True

            # Update improvement counter
            if no_improvement:
                self.no_improvement_counter += 1
            else:
                self.no_improvement_counter = 0

            # Check if improvement 
            if self.no_improvement_counter >= self.no_improvement_limit:
                self.stop_reason = "No improvement limit reached"
                return True
        
        # update state
        self.prev_error = self.mean_err
        self.iteration += 1

        return False
    


class IterativeClosestPoint():
    '''
    Implements the point-to-plane ICP algorithm for scan matching. The main method is 'find_transformation', which
    takes two point clouds as input and returns the transformation parameters that best align the new data points to
    the true data points. The class also includes methods for outlier rejection, normal computation, and information
    storage for analysis and debugging.
    '''
    IDX_X= 0
    IDX_Y= 1
    IDX_THETA= 2
    MIN_POINTS = 3
    EPSILON = 1e-9

    def __init__(self, stop_params: dict, max_n_points:int=800, max_correspondence_distance= 2.0, neighbors: int = 10):
        '''
        Initializes the ICP scan matcher with the given stop parameters and maximum correspondence distance.

        Parameters:
        ----------
            stop_params: A dictionary containing the parameters for the ICP stop condition. This includes:
                - max_iterations: Maximum number of ICP iterations before stopping.
                - epsilon_rel: Minimum relative improvement required to continue ICP.
                - no_improvement_limit: Number of consecutive iterations with insufficient improvement before stopping.
                - min_error: Minimum mean error threshold for stopping.
                - min_dtrans: Minimum translation change required to continue ICP.
                - min_drot: Minimum rotation change required to continue ICP.
            max_correspondence_distance: The maximum distance between corresponding points to be considered in the ICP
            neighbors_pca: The number of neighbors to use for PCA when computing normals.
    
        '''    
        # Init NN
        self.neighbor= NearestNeighbors(n_neighbors=neighbors, algorithm='kd_tree')        
        
        # Init params
        # Max dist for correspondences. All correspondences with a bigger distance will be rejected as outliers
        self.max_correspond_dist= max_correspondence_distance
        # Number of neighbors to use for PCA when computing normals
        self.neighbors = neighbors
        # The true pointclous data will be subsampled to this amount, in every run (before outlier rejection, etc) 
        self.max_n_points = max_n_points

        # Init numpys random generator
        self.rng = np.random.default_rng()

        # Init points count
        self.n_points_true_data = None
        self.n_points_new_data = None

        # Init stop condition
        self.stop_condition = ICPStopCondition(
            max_iterations=stop_params.get("max_iterations", 10),
            epsilon_rel=stop_params.get("epsilon_rel", 1e-3),
            no_improvement_limit=stop_params.get("no_improvement_limit", 2),
            min_error=stop_params.get("min_error", 1.0),
            min_dtrans=stop_params.get("min_dtrans", 1e-4),
            min_drot=stop_params.get("min_drot", 1e-1)
        )

        # Add info storage
        self.info: dict = {}

        # Store infos in members
        self.transformed_new_data_list = []
        self.squared_error_list = []
        self.transformation_parameter_list = []
        self.list_of_cleaned_corresp = []
        self.list_of_cleaned_corresp_numb = []
        self.list_of_corresp_numb = []


    def store_info(self, extended: bool = False):
        '''
        Stores the current state of the ICP stop condition and other relevant information in the 'info' member variable.
        '''
        self.stop_condition.store_info()
        self.info = self.stop_condition.get_info()
        self.info["max_correspondence_distance"] = self.max_correspond_dist
        self.info["n_points_true_data"] = self.n_points_true_data
        self.info["n_points_new_data"] = self.n_points_new_data

        if extended:
            self.info["transformed_new_data_list"] = self.transformed_new_data_list
            self.info["squared_error_list"] = self.squared_error_list
            self.info["transformation_parameter_list"] = self.transformation_parameter_list
            self.info["list_of_cleaned_corresp"] = self.list_of_cleaned_corresp
            self.info["list_of_cleaned_corresp_numb"] = self.list_of_cleaned_corresp_numb
            self.info["list_of_corresp_numb"] = self.list_of_corresp_numb


    def get_info(self) -> dict:
        '''
        Returns the current state of the ICP stop condition and other relevant information.

        Returns
        ----------
            dict: With the following structure:
            {
                "iteration": int,
                "mean_err": float,
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
                "list_of_cleaned_corresp_numb": List[int]  # List of number of cleaned correspondences at each iteration
                "list_of_corresp_numb": List[int]  # List of number of correspondences at each iteration
            }
        '''
        return self.info


    @staticmethod
    def sanitize_pointcloud(pointcloud: np.ndarray) -> np.ndarray:
        '''Return a finite Nx2 pointcloud array. Invalid rows are removed.''' 
        pointcloud = np.asarray(pointcloud, dtype=float)

        if pointcloud.ndim != 2 or pointcloud.shape[1] != 2:
            return np.empty((0, 2), dtype=float)

        finite_rows = np.all(np.isfinite(pointcloud), axis=1)
        return pointcloud[finite_rows]


    @staticmethod
    def correct_pose(pose:Tuple[float, float, float] , transf_param: np.ndarray) -> Tuple[float, float, float]:
        '''
        Corrects the given pose by the given transformation.
        '''
        tx, ty, rot_theta = transf_param.flatten()
        x, y, theta = pose

        # Compute sin, cos
        c = np.cos(rot_theta)
        s = np.sin(rot_theta)

        # Define transformation matrix 
        T = np.array([
            [c, -s, tx],
            [s, c, ty],
            [0, 0, 1],
        ])

        # get point
        p = np.array([x, y, 1])

        # Transform point and theta
        p_new = T @ p
        theta_new = theta + rot_theta

        # normalize angle
        theta_new = atan2(sin(theta_new), cos(theta_new))

        return (p_new[0], p_new[1], theta_new)


    @staticmethod
    def compute_normals(points: np.ndarray, step: int = 1) -> np.ndarray:
        '''Gets a numpy array of points and and a step value and calculates the corresponding
        normal vectors for every point in the given array.'''
        if points.shape[0] == 0:
            return []

        normals = [np.array([0.0, 0.0])]
        visualize_normals = []
        IDX_X= 0
        IDX_Y= 1
        
        for i in range(step, points.shape[0] - step):
            # Choose points
            previous_point= points[i-step]
            current_point= points[i]
            next_point= points[i+step]
        
            # Compute normal vector
            vector= next_point - previous_point
            vector_norm = np.linalg.norm(vector)

            if not np.isfinite(vector_norm) or vector_norm <= IterativeClosestPoint.EPSILON:
                normal = np.array([0.0, 0.0])
            else:
                unit_vector= vector / vector_norm
                normal= np.array([-unit_vector[IDX_Y], unit_vector[IDX_X]])

            normals.append(normal)
        
            # For Visualization
            point_plus_normal= current_point + normal
            visualize_normal= (current_point, point_plus_normal)
            visualize_normals.append(visualize_normal)
        
        normals.append(np.array([0.0, 0.0]))
        return normals


    @staticmethod
    def compute_normals_knn_pca(points: np.ndarray, k: int = 10) -> np.ndarray:
        """
        Compute normals using KNN + PCA (2D). Find the k NN for each point. Than we find the direction of the lowest variance 
        for that points by PCA. This direction is the normal direction, the direction perpendicular to the local surface.
        
        Parameters:
        ----------
            points: Nx2 numpy array of points.
            k: Number of neighbors to use for normal estimation.
        
        Returns:
        ----------
            normals: Nx2 numpy array of normal vectors corresponding to each point.
        """
        # Check if we have enough points
        n_points = points.shape[0]
        if n_points < k:
            return np.zeros((n_points, 2))

        # init KNN and find neighbors
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(points)
        _, indices = nbrs.kneighbors(points)

        normals = np.zeros((n_points, 2))

        # For each point, compute normal based on direction with lowest variance (ICP)
        for i in range(n_points):
            # get k neighbors
            neighbors = points[indices[i]]

            # Center neighbors
            centroid = np.mean(neighbors, axis=0)
            centered = neighbors - centroid

            # Covariance matrix (2x2)
            cov = centered.T @ centered

            # Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(cov)

            # Smallest eigenvector = normal direction
            normal = eigvecs[:, 0]

            # Normalize
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                normal = normal / norm
            else:
                normal = np.array([0.0, 0.0])

            normals[i] = normal

        return normals


    def compute_normals_pca(self, points: np.ndarray, k: int = 10):
        # Check if we have enough points
        n_points = points.shape[0]
        if n_points < k:
            return np.zeros((n_points, 2))
        
        _, indices = self.neighbor.kneighbors(points, n_neighbors=k)

        normals = np.zeros((n_points, 2))

        # For each point, compute normal based on direction with lowest variance (ICP)
        for i in range(n_points):
            # get k neighbors
            neighbors = points[indices[i]]

            # Center neighbors
            centroid = np.mean(neighbors, axis=0)
            centered = neighbors - centroid

            # Covariance matrix (2x2)
            cov = centered.T @ centered

            # Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(cov)

            # Smallest eigenvector = lowest variance = normal direction
            normal = eigvecs[:, 0]

            # Normalize vectors
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                normal = normal / norm
            else:
                normal = np.array([0.0, 0.0])

            normals[i] = normal

        return normals


    @staticmethod
    def compute_rotation_matrix(theta: float) -> np.ndarray:
        '''Return rotation matrix of given theta.'''
        theta = float(np.asarray(theta).item())
        return np.array([
            [cos(theta), -sin(theta)],
            [sin(theta), cos(theta)]
            ])


    def max_distance_outlier_rejection(self, new_data_points: np.ndarray, true_data_points: np.ndarray, correspondences: list) -> tuple:
        '''Get's a list of new data points and true data points and a list of (i, j) correspondences. 
        Rejects all pairs, which distance is higher than the given threshold.'''
        cleaned_correspondences= []
        sum_error= 0
        
        for i, j in correspondences:
            error= np.linalg.norm(new_data_points[i] - true_data_points[j])

            if not np.isfinite(error):
                continue

            sum_error+= error
            if error < self.max_correspond_dist:
                cleaned_correspondences.append((i, j))

        return cleaned_correspondences, sum_error


    @staticmethod
    def multiple_pairing_rejection(correspondences: list) -> list:
        '''Get's a sorted list of (j, i, distance) correspondences, sorted by j and rejects the 
        worst (j, i) correspondences, such that there is only one i that belongs to one j. One correspon-
        dence is more worse than the other, when the distance between the corresponding points is bigger, 
        than the other. returns a list of (i, j) correspondences.'''
        if not correspondences:
            return []

        # Pop first item 
        j, i, dist= heappop(correspondences)
        current_j= j

        # init all variables
        cleaned_correspondences= []
        c=(i, j)
        shortest_dist= dist

        # Search for best pairs
        for i in range(len(correspondences)):
            j, i ,dist = heappop(correspondences)
            if(j != current_j):
                current_j = j
                cleaned_correspondences.append(c)
                shortest_dist = inf
                c = (i, j)
            elif(dist < shortest_dist):
                c = (i, j)
        
        cleaned_correspondences.append(c)
        
        return cleaned_correspondences

    
    def outlier_rejection(self, new_data_points: np.ndarray, true_data_points: np.ndarray, correspondences: list) -> tuple[np.ndarray, float]:
        '''Class that uses all available methodes to reject outliers in the (j, i, distance)
        correspondences.'''
        # print("\n\nNumber Of correspondences before mpr= ", len(correspondences))
        cleaned_correspondences= self.multiple_pairing_rejection(correspondences)
        # print("Number Of correspondences after mpr= ", len(cleaned_correspondences))
        cleaned_correspondences, sum_error= self.max_distance_outlier_rejection(new_data_points, true_data_points, cleaned_correspondences)
        # print("Number Of correspondences after max_distance_outlier_rejection= ", len(cleaned_correspondences))
        return cleaned_correspondences, sum_error


    @staticmethod
    def compute_jacobian_point_to_plane(normal: np.ndarray, theta: float, point: np.ndarray) -> np.ndarray:        
        theta = float(np.asarray(theta).item())
        x= point.item(0)
        y= point.item(1)
        x_normal= normal.item(0)
        y_normal= normal.item(1)
        third_element= x_normal * (-x*sin(theta) - y*cos(theta)) + y_normal * (x*cos(theta) - y*sin(theta))
        return np.array([[x_normal, y_normal, third_element]])


    def prepare_system_point_to_plane(
            self, 
            transformation_parameter: np.ndarray,
            latest_new_data: np.ndarray,
            true_data_pointpairs: np.ndarray,
            correspondences: list,
            true_data_normals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        '''
        Prepares the system of equations for the point-to-plane ICP algorithm. Checks if the inputs are valid and computes the Hessian
        matrix H and the gradient vector g for the least squares problem. Also computes the squared error (point to plane) for the
        current correspondences.
        '''
        # Init Hessian Matrix and gradient
        H = np.zeros((3, 3))
        g = np.zeros((3, 1))        
        squared_error= 0.0

        for i, j in correspondences:
            # Extract data
            new_data_point= latest_new_data[i]
            true_data_point= true_data_pointpairs[j]
            normal= true_data_normals[j]

            # Check if data is valid
            if not (
                np.all(np.isfinite(new_data_point)) and
                np.all(np.isfinite(true_data_point)) and
                np.all(np.isfinite(normal))
            ):
                continue

            if np.linalg.norm(normal) <= self.EPSILON:
                continue
            
            # Compute the distance error between the transformed new point and the true point
            distance_error= new_data_point - true_data_point            
            
            # Compute point to plane error by projecting the distance error onto the normal vector
            normal_error= np.dot(normal, distance_error)
            if not np.isfinite(normal_error):
                continue
            
            # : Weight the correspondence by error
            weight = 1 / (1 + normal_error**2)
            
            # Compute jacobian matrix
            J= self.compute_jacobian_point_to_plane(normal, transformation_parameter[self.IDX_THETA], new_data_point)

            if not np.all(np.isfinite(J)):
                continue
            
            # Update Hessian and gradient
            H+= weight * np.dot(J.T, J)
            g+= weight * np.dot(J.T, normal_error)   
            
            # Accumulate the squared errors
            squared_error+= normal_error**2

        return H, g, squared_error


    def downsample_pointcloud_rand(self, pointcloud: np.ndarray, max_n_points: int=800) -> np.ndarray:
        '''
        Downsamples the given pointcloud to the given max number of points, randomly.
        '''
        n_points = pointcloud.shape[0]

        if n_points >= max_n_points:
            indices = self.rng.choice(n_points, size=max_n_points, replace=False)
            subsampled_pointcloud = pointcloud[indices]
        else:
            subsampled_pointcloud = pointcloud
        
        return subsampled_pointcloud


    def dowmsample_pointcloud_deterministic(self, pointcloud: np.ndarray, max_n_points: int=800) -> np.ndarray:
        '''
        Downsamples the given pointcloud to the given max number of points, deterministically.
        The selected points are approximately evenly spaced over the original order.
        '''
        pointcloud = np.asarray(pointcloud, dtype=float)

        if pointcloud.ndim != 2 or pointcloud.shape[1] != 2:
            return np.empty((0, 2), dtype=float)

        n_points = pointcloud.shape[0]

        if n_points <= max_n_points:
            return pointcloud

        # Deterministic evenly spaced sampling
        indices = np.linspace(0, n_points - 1, max_n_points, dtype=int)
        subsampled_pointcloud = pointcloud[indices]

        return subsampled_pointcloud


    def find_transformation(
            self, 
            new_data_pointpairs: np.ndarray, 
            true_data_pointpairs: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        '''
        Get's the new data points and the true datapoints and trys to minimize the error between the two
        pointclouds by finding the best transformation. Returns the transformation parameters and stores 
        relevant information in the 'info' member variable. The info contains the data for each transformation
        run.

        Parameters:
        ----------
            new_data_pointpairs: Nx2 numpy array of the new data points.
            true_data_pointpairs: Mx2 numpy array of the true data points.
        
        Returns:
        ----------
            transformation_parameter: 3x1 numpy array of the transformation parameters (tx, ty, theta).
        '''
        # Santize pointclouds
        new_data_pointpairs = self.sanitize_pointcloud(new_data_pointpairs)
        true_data_pointpairs = self.sanitize_pointcloud(true_data_pointpairs)

        # Init vars
        transformation_parameter = np.zeros((3, 1))
        dtransformation = transformation_parameter.copy()
        squared_error = inf
        mean_err = inf  

        if (
            new_data_pointpairs.shape[0] < self.MIN_POINTS or
            true_data_pointpairs.shape[0] < self.MIN_POINTS
        ):
            return transformation_parameter, [new_data_pointpairs.copy()], [], []

        # Store number of points for logging
        self.n_points_true_data = true_data_pointpairs.shape[0]
        self.n_points_new_data = new_data_pointpairs.shape[0]

        # Downsample true data points
        true_data_pointpairs = self.dowmsample_pointcloud_deterministic(
            pointcloud=true_data_pointpairs,
            max_n_points=self.max_n_points
        )

        # List to save results
        self.squared_error_list= []
        self.transformation_parameter_list = [transformation_parameter.copy()]
        self.transformed_new_data_list = [new_data_pointpairs.copy()]
        latest_new_data= new_data_pointpairs.copy()
        self.list_of_cleaned_corresp = []
        self.list_of_cleaned_corresp_numb = []
        self.list_of_corresp_numb = []
        
        # Train Nearest Neighbor with true data points 
        self.neighbor.fit(true_data_pointpairs)
        
        # Compute normals of true data points
        # true_data_normals= self.compute_normals(true_data_pointpairs)
        # true_data_normals = self.compute_normals_knn_pca(
        #     points=true_data_pointpairs,
        #     k=self.neighbors
        # )

        true_data_normals = self.compute_normals_pca(
            points=true_data_pointpairs,
            k=self.neighbors
        )
                
        # Variable to save squared error. 
        sum_error= 10**10
        
        while not self.stop_condition.stop_icp(mean_err, dtransformation):
            # Find Nearest Neighbor by euclidean distance
            correspondences= []

            if latest_new_data.shape[0] == 0:
                break

            distances, indices = self.neighbor.kneighbors(latest_new_data, n_neighbors=1)
            for i in range(np.shape(latest_new_data)[0]):
                # Push correspondences to heap, sorted by the index of the true data pointcloud j
                heappush(correspondences, (indices.item(i), i, distances[i]))
            
            # Outlier Rejection            
            cleaned_correspondences, sum_error = self.outlier_rejection(latest_new_data, true_data_pointpairs, correspondences)

            if len(cleaned_correspondences) < self.MIN_POINTS:
                break
            
            # Prepare the system
            H, g, squared_error = self.prepare_system_point_to_plane(
                transformation_parameter,
                latest_new_data,
                true_data_pointpairs,
                cleaned_correspondences,
                true_data_normals,
            )

            # Check if Hessian and g are finite
            if not (np.all(np.isfinite(H)) and np.all(np.isfinite(g))):
                break
            
            # Compute least Squares Solution
            dtransformation= np.linalg.lstsq(H, -g, rcond=None)[0]

            if not np.all(np.isfinite(dtransformation)):
                break
            
            # Update transformation parameter 
            transformation_parameter+= dtransformation
            
            # Ensure valid angle
            theta = float(transformation_parameter[self.IDX_THETA].item())
            transformation_parameter[self.IDX_THETA] = atan2(sin(theta), cos(theta))
            
            # Update rotation and translation matrix
            rotation_matrix= self.compute_rotation_matrix(transformation_parameter[self.IDX_THETA])
            translation= transformation_parameter[0:self.IDX_THETA]
            
            # Transform new data points by rotation and translation 
            latest_new_data_T= np.dot(rotation_matrix, new_data_pointpairs.T) + translation
            latest_new_data= latest_new_data_T.T

            # Compute mean err metric for stop condition check
            if np.isfinite(squared_error):
                mean_err = squared_error / len(cleaned_correspondences)
            else:
                mean_err = inf
            
            # Append data to lists
            self.transformed_new_data_list.append(latest_new_data)
            self.list_of_cleaned_corresp.append(cleaned_correspondences)
            self.list_of_cleaned_corresp_numb.append(len(cleaned_correspondences))
            self.list_of_corresp_numb.append(len(correspondences))
            self.squared_error_list.append(squared_error)
            self.transformation_parameter_list.append(transformation_parameter.copy())
        
        if self.list_of_cleaned_corresp:
            self.list_of_cleaned_corresp.append(self.list_of_cleaned_corresp[-1])
            self.list_of_cleaned_corresp_numb.append(self.list_of_cleaned_corresp_numb[-1])
            self.list_of_corresp_numb.append(self.list_of_corresp_numb[-1])

        # Store info and reset stop condition for next run
        self.store_info(extended=True)
        self.stop_condition.reset()
        
        # Print stop reasons
        # icp_stop_info = self.get_info()
        # print("\n\nICP Stop Condition Info:")
        # keys = ['iteration', 'mean_err', 'rel_improvement', 'no_improvement_counter', 'dtrans_norm', 'drot_abs', 'stop_reason']
        # for key in keys:
        #     print(f"{key}: {icp_stop_info[key]}")

        return transformation_parameter

