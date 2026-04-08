import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, atan2, pi
import time as t
from sklearn.neighbors import NearestNeighbors
from heapq import heappush, heappop


class IterativeClosestPoint():
    IDX_X= 0
    IDX_Y= 1
    IDX_THETA= 2
    def __init__(self, max_number_of_iterations= 10, max_correspondence_distance= 2.0):
        # Max allowed number of iterations
        self.max_number_of_iterations= max_number_of_iterations
        
        # Max number of correspondences
        self.max_correspondence_distance= max_correspondence_distance
        
        # Init Nearest Neighbor
        self.neighbor= NearestNeighbors(n_neighbors=1, algorithm='kd_tree')        
        self.min_squared_error= 15.0

        # Init numpys random generator
        self.rng = np.random.default_rng()


    @staticmethod
    def compute_normals(points, step= 1):
        '''Gets a numpy array of points and and a step value and calculates the corresponding
        normal vectors for every point in the given array.'''
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
            unit_vector= vector/np.linalg.norm(vector)
            normal= np.array([-unit_vector[IDX_Y], unit_vector[IDX_X]])
            normals.append(normal)
        
            # For Visualization
            point_plus_normal= current_point + normal
            visualize_normal= (current_point, point_plus_normal)
            visualize_normals.append(visualize_normal)
        
        normals.append(np.array([0.0, 0.0]))
        return normals


    @staticmethod
    def compute_rotation_matrix(theta):
        '''Return rotation matrix of given theta.'''
        return np.array([
            [cos(theta), -sin(theta)],
            [sin(theta), cos(theta)]
            ])


    def max_distance_outlier_rejection(self, new_data_points, true_data_points, correspondences):
        '''Get's a list of new data points and true data points and a list of (i, j) correspondences. 
        Rejects all pairs, which distance is higher than the given threshold.'''
        cleaned_correspondences= []
        sum_error= 0
        
        for i, j in correspondences:
            error= np.linalg.norm(new_data_points[i] - true_data_points[j])
            sum_error+= error
            # error= new_data_points[i] - true_data_points[j]
            if(np.linalg.norm(error) < self.max_correspondence_distance):    
                cleaned_correspondences.append((i, j))
        
        print("Error is ", sum_error)
        return cleaned_correspondences, sum_error


    @staticmethod
    def multiple_pairing_rejection(correspondences):
        '''Get's a sorted list of (j, i, distance) correspondences, sorted by j and rejects the 
        worst (j, i) correspondences, such that there is only one i that belongs to one j. One correspon-
        dence is more worse than the other, when the distance between the corresponding points is bigger, 
        than the other. returns a list of (i, j) correspondences.'''
        # Pop first item 
        j, i, dist= heappop(correspondences)
        current_j= j
        # init all variables
        cleaned_correspondences= []
        c=(i, j)
        shortest_dist= dist
        # Search for best pairs
        for i in range(len(correspondences)):
            j, i ,dist= heappop(correspondences)
            if(j != current_j):
                current_j= j
                cleaned_correspondences.append(c)
                shortest_dist= 10**10
                c= (i, j)
            elif(dist < shortest_dist):
                c= (i, j)
        cleaned_correspondences.append(c)
        return cleaned_correspondences

    
    def outlier_rejection(self, new_data_points, true_data_points, correspondences):
        '''Class that uses all available methodes to reject outliers in the (j, i, distance)
        correspondences.'''
        # print("\n\nNumber Of correspondences before mpr= ", len(correspondences))
        cleaned_correspondences= self.multiple_pairing_rejection(correspondences)
        # print("Number Of correspondences after mpr= ", len(cleaned_correspondences))
        cleaned_correspondences, sum_error= self.max_distance_outlier_rejection(new_data_points, true_data_points, cleaned_correspondences)
        # print("Number Of correspondences after max_distance_outlier_rejection= ", len(cleaned_correspondences))
        return cleaned_correspondences, sum_error


    @staticmethod
    def compute_jacobian_point_to_plane(normal, theta, point):        
        x= point.item(0)
        y= point.item(1)
        x_normal= normal.item(0)
        y_normal= normal.item(1)
        third_element= x_normal * (-x*sin(theta) - y*cos(theta)) + y_normal * (x*cos(theta) - y*sin(theta))
        return np.array([[x_normal, y_normal, third_element]])


    def prepare_system_point_to_plane(self, transformation_parameter, latest_new_data, true_data_pointpairs, correspondences, true_data_normals):
        # Init Hessian Matrix and gradient
        H = np.zeros((3, 3))
        g = np.zeros((3, 1))        
        squared_error= 0.0
        for i, j in correspondences:
            new_data_point= latest_new_data[i]
            true_data_point= true_data_pointpairs[j]
            normal= true_data_normals[j]
            
            # Compute the distance error between the transformed new point and the true point
            distance_error= new_data_point - true_data_point            
            
            # Compute normal error
            normal_error= np.dot(normal, distance_error)
            
            # : Weight the correspondence by error
            weight = 1 / (1 + normal_error**2)
            
            # Compute jacobian matrix
            J= self.compute_jacobian_point_to_plane(normal, transformation_parameter[self.IDX_THETA], new_data_point)
            
            # Update Hessian and gradient
            H+= weight * np.dot(J.T, J)
            g+= weight * np.dot(J.T, normal_error)   
            
            # Accumulate the squared errors
            squared_error+= normal_error**2
        return H, g, squared_error


    @staticmethod
    def downsample_pointcloud(self, pointcloud: np.ndarray, max_n_points: int=800):
        '''
        Downsamples the given pointcloud to the given max numbber of points, randomly.
        '''
        n_points = pointcloud.shape[0]

        if n_points >= max_n_points:
            indices = self.rng.choice(n_points, size=max_n_points, replace=False)
        
        return n_points[indices]


    def find_transformation(self, new_data_pointpairs, true_data_pointpairs):
        # Downsample true data points
        true_data_pointpairs = self.downsample_pointcloud(
            pointcloud=true_data_pointpairs,
            max_n_points=800
        )

        # Transformation parameter vector (tx, ty, theta)
        transformation_parameter = np.zeros((3, 1))
        
        # List to save results
        squared_error_list= []
        transformation_parameter_list= [transformation_parameter.copy()]
        transformed_new_data_list= [new_data_pointpairs.copy()]
        latest_new_data= new_data_pointpairs.copy()
        list_of_correspondences= []
        
        # Train Nearest Neighbor with true data points 
        self.neighbor.fit(true_data_pointpairs)
        
        # Compute normals of true data points
        true_data_normals= self.compute_normals(true_data_pointpairs)
        
        # Calculate transformation iteratively
        index= 0
        
        # Variable to save squared error. 
        sum_error= 10**10
        
        while(index < self.max_number_of_iterations and sum_error > self.min_squared_error):
            index+=1
            
            # Find Nearest Neighbor by euclidean distance
            correspondences= []
            distances, indices= self.neighbor.kneighbors(latest_new_data)
            for i in range(np.shape(latest_new_data)[0]):
                # Push correspondences to heap, sorted by the index of the true data pointcloud j
                heappush(correspondences, (indices.item(i), i, distances[i]))
            
            # Outlier Rejection            
            cleaned_correspondences, sum_error= self.outlier_rejection(latest_new_data, true_data_pointpairs, correspondences)            
            
            # Prepare the system
            H, g, squared_error= self.prepare_system_point_to_plane(transformation_parameter, latest_new_data, true_data_pointpairs, cleaned_correspondences, true_data_normals)
            
            # Compute least Squares Solution
            dtransformation= np.linalg.lstsq(H, -g, rcond=None)[0]
            
            # Update transformation parameter 
            transformation_parameter+= dtransformation
            
            # Ensure valid angle
            transformation_parameter[self.IDX_THETA]= atan2(sin(transformation_parameter[self.IDX_THETA]), cos(transformation_parameter[self.IDX_THETA]))            
            
            # Update rotation and translation matrix
            rotation_matrix= self.compute_rotation_matrix(transformation_parameter[self.IDX_THETA])
            translation= transformation_parameter[0:self.IDX_THETA]
            
            # Transform new data points by rotation and translation 
            latest_new_data_T= np.dot(rotation_matrix, new_data_pointpairs.T) + translation
            latest_new_data= latest_new_data_T.T
            
            # Append data to lists
            transformed_new_data_list.append(latest_new_data)
            list_of_correspondences.append(cleaned_correspondences)
            squared_error_list.append(squared_error)
            transformation_parameter_list.append(transformation_parameter.copy())
        
        list_of_correspondences.append(list_of_correspondences[-1])
        
        return transformation_parameter, transformed_new_data_list, squared_error_list, cleaned_correspondences

