#!/usr/bin/env python3

import rospy
# Python libarys
import threading
import numpy as np
from math import atan2, sin, cos, pi, sqrt, radians, degrees, isfinite, floor
import random
from scipy.stats import norm as normal_dist
from sklearn.neighbors import NearestNeighbors 
# Messages and Services
from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion
from sensor_msgs.msg import LaserScan
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler
# Custom messages 
from rvc_commander.msg import WheelEncoder
from rvc_commander.msg import Float64Array


'''
Description:

    Monte Carlo Localization Algorithm with uses a pre computed likelihood field as the measurement model. Unfortuanetly the pre computation
    takes way too long. With the nearest neighbors methode of sklearn it takes (45 to 48 seconds) which is not feasible.

'''


#__________________________________________________________________________________________________________________________________
# Particle Filter Implementation  
#__________________________________________________________________________________________________________________________________

class ParticleFilter():
    def __init__(self, particle_filter_parameter, start_pose, map_parameter_particle_filter, occupancy_values, occupancy_grid_map): 
        robot_parameter, uncertainty_parameter, measurement_parameter= particle_filter_parameter
        # Robot parameter
        wheel_saperation= robot_parameter
        # Uncertainty parameter
        self.control_motion_factor, self.control_turn_factor, self.measurement_distance_stddev, self.measurement_angle_stddev= uncertainty_parameter
        # Map parameter
        map_width_m, map_height_m, self.grid_resolution= map_parameter_particle_filter
        self.map_width= int(map_width_m / self.grid_resolution)
        self.map_height= int(map_height_m / self.grid_resolution)
        self.shift_x= map_width_m / 2
        self.shift_y= map_height_m / 2
        self.occ, self.free, self.unknown= occupancy_values
        # Occupancy grid map
        self.occupancy_grid_map= occupancy_grid_map
        # Initialize particles.
        self.particles= None
        self.initialize_particles(start_pose, map_width_m, map_height_m)
        # Robot parameter.
        self.wheel_saperation= wheel_saperation
        # Measurement parameter
        self.min_sensor_range, self.max_sensor_range= measurement_parameter
        # Resampling 
        number_of_particles= len(self.particles)
        self.neff_threshold= number_of_particles * (1/2)
        self.weights= []
        self.initialize_weights()
        # Test variables____________________________________________________________________________
        self.flag= True
        self.normalized_weights= None
        

    def initialize_particles(self, start_pose, map_width_m, map_height_m):
        '''
        Depends on if the start pose of the robot is known or not.'''
        self.particles= []
        '''
        # 1) Start pose not known
        # Standarddeviation
        standard_deviation= (map_width_m/2, map_height_m/2, pi)
        # One particle per m^2
        number_of_particles= int(map_width_m * map_height_m)
        # Initialize particles
        for i in range(number_of_particles):
            self.particles.append([
                random.gauss(start_pose[j], standard_deviation[j]) for j in range(3)])
        '''
        # 2)Assuming the robot starts in his docking station (known start pose)
        standard_deviation= (0.5, 0.5, 10*pi/180)
        number_of_particles= 70
        for i in range(number_of_particles):
            particle= [ random.gauss(start_pose[j], standard_deviation[j]) for j in range(3)]
            self.particles.append(particle)


    #____________________________________________________________________________________________________________________
    # Prediction 
    #____________________________________________________________________________________________________________________
    @staticmethod
    def g(state, control, wheel_saperation):
        '''Motion Model for differential drive robot.'''
        x, y, theta = state
        distance_left_wheel, distance_right_wheel = control
        if distance_right_wheel != distance_left_wheel:
            alpha = (distance_right_wheel - distance_left_wheel) / wheel_saperation
            rad = distance_left_wheel/alpha
            g1 = x + (rad + wheel_saperation/2.)*(sin(theta+alpha) - sin(theta))
            g2 = y + (rad + wheel_saperation/2.)*(-cos(theta+alpha) + cos(theta))
            g3 = (theta + alpha + pi) % (2*pi) - pi
        else:
            g1 = x + distance_left_wheel * cos(theta)
            g2 = y + distance_left_wheel * sin(theta)
            g3 = theta
        return (g1, g2, g3)


    def predict(self, control):
        """The prediction step of the particle filter."""
        # Calculate left and right control standarddeviation.
        left_control, right_control = control
        control_difference= left_control - right_control
        control_turn_variance= (self.control_turn_factor * control_difference)**2
        left_control_variance= (self.control_motion_factor * left_control)**2 + control_turn_variance
        right_control_variance= (self.control_motion_factor * right_control)**2 + control_turn_variance
        left_control_stddv= sqrt(left_control_variance)
        right_control_stddv= sqrt(right_control_variance)
        # Sample control values and calculate new particle pose.
        for i in range(len(self.particles)):
            sampled_left_control= random.gauss(left_control, left_control_stddv)
            sampled_right_control= random.gauss(right_control, right_control_stddv)
            self.particles[i]= ParticleFilter.g(self.particles[i], [sampled_left_control, sampled_right_control], self.wheel_saperation) 


    #____________________________________________________________________________________________________________________
    # Correction 
    #____________________________________________________________________________________________________________________

    def find_occupied_cells(self):
        '''Finds all occupied cells in the occupancy grid map. Returns a list of tuples, where each tuple
        contains a (x, y) grid cell position tuple and a (i, j) grid cell indices tuple.'''
        occupied_cells= []
        rows, columns= np.shape(self.occupancy_grid_map)
        for i in range(rows):
            for j in range(columns):
                if(self.occupancy_grid_map[i][j] == self.occ):
                    x, y= self.transform_grid_cell_to_point((i, j))
                    occupied_cells.append([(x, y), (i, j)])
        return occupied_cells


    @staticmethod
    def find_nearest_neighbor(point, list_of_points):
        '''Gets a point and a list of points. Finds the closest point in the list of points. Returns
        a list of tuples where each tuple consits of the shortest point and the distance. A list is 
        necessary, because there could be multiple points which have the same distance.'''
        nearest_neighbors= []
        squared_distance= 0
        shortest_distance= 10**6
        if(list_of_points):
            # Find nearest neighbor 
            for p in list_of_points:
                # Calculate euclidean distance
                for i in range(len(point)):
                    deltax= point[i] - p[i]
                    squared_distance+= deltax **2                
                distance= sqrt(squared_distance)
                squared_distance= 0                
                # Check if distance is shorter than the distances before -> initialize new distance 
                if(distance < shortest_distance):
                    shortest_distance= distance
                    nearest_neighbors= [(p, distance)]
                # If new distance equal shorttest distance -> append 
                elif(distance == shortest_distance):
                    nearest_neighbors.append((p, distance))
        return nearest_neighbors
    
    
    def compute_likelihood_field(self):
        '''Pre computes the likelihood field of the map.'''
        # Create empty likelihood field
        rows, columns= np.shape(self.occupancy_grid_map)
        self.occupancy_field= np.ones((rows, columns))
        # Find all occupied cells
        occupied_cells= self.find_occupied_cells()
        occupied_cell_points= [point for point, indices in occupied_cells]
        rospy.loginfo("Number of occupied cells: %i", len(occupied_cell_points))
        # Compute likelihood field
        for i in range(int(rows)):
            for j in range(int(columns)):
                cell_point= self.transform_grid_cell_to_point((i, j))
                nearest_neighbor= self.find_nearest_neighbor(cell_point, occupied_cell_points)
                # if(len(nearest_neighbor) >= 2):
                #     rospy.loginfo("Found multiple neighbors.")
    

    def compute_likelihood_field_with_NearestNeighbors(self):
        '''Pre computes the likelihood field of the map.'''
        # Create empty likelihood field
        rows, columns= np.shape(self.occupancy_grid_map)
        self.occupancy_field= np.ones((rows, columns))
        # Find all occupied cells
        occupied_cells= self.find_occupied_cells()
        occupied_cell_points= [point for point, indices in occupied_cells]
        rospy.loginfo("Number of occupied cells: %i", len(occupied_cell_points))
        # Initialize Nearest Neighbors
        neighbor= NearestNeighbors(n_neighbors=1, algorithm='brute')
        neighbor.fit(occupied_cell_points)
        # Compute likelihood field
        for i in range(int(rows)):
            for j in range(int(columns)):
                cell_point= self.transform_grid_cell_to_point((i, j))
                nearest_occupied_cell= neighbor.kneighbors([cell_point])


    def initialize_weights(self):
        '''Initialize the weights by one.''' 
        number_of_weights= len(self.particles)
        self.weights= [1.0 for i in range(number_of_weights)]


    def compute_weights(self, measurements):
        '''Compute a weight for each particle depending on the measurements.'''
        index= 0


    @staticmethod
    def number_of_effective_particles(weights):
        '''Calculate the number of effective weights.''' 
        sum_of_squared_weights= 0
        for w in weights:
            sum_of_squared_weights+= w**2
        return 1/sum_of_squared_weights


    def low_variance_sampler(self, weights):
        '''Stochastic universal resampling without saving the accumulated weights in an array.'''
        number_of_weights= len(weights)
        acc_weight= weights[0]
        new_particles= []
        particle_index= 0
        # Pick particles according to weight.
        random_number= random.uniform(0.0, 1/number_of_weights)
        for j in range(number_of_weights):
            u= random_number + j * (1/number_of_weights)
            while(u > acc_weight):
                particle_index+= 1
                acc_weight+= weights[particle_index]                    # List index out of range
            new_particles.append(self.particles[particle_index])
        # return new_particles.copy()
        return new_particles


    def correct(self, measurements):
        # self.normalized_weights= self.compute_weights(measurements)
        # self.particles= self.low_variance_sampler(self.normalized_weights)
         # First compute all weights.
        self.compute_weights(measurements)
        # Decide if resampling is needed based on number of effective particles
        if(self.number_of_effective_particles(self.weights) < self.neff_threshold):
            self.particles = self.low_variance_sampler(self.weights)
            # Reinitialize weights
            self.initialize_weights()
            rospy.loginfo("Resampling")
        else: 
            rospy.loginfo("No resampling ")
 

    #____________________________________________________________________________________________________________________
    # Helper Functions 
    #____________________________________________________________________________________________________________________

    def transform_point_to_grid_cell(self, point):
        '''Transforms an (x, y) point to the array access indices (i, j for row, column). '''
        x,y = point
        x_shifted= x + self.shift_x
        y_shifted= y + self.shift_y
        i= floor(y_shifted/self.grid_resolution)
        j= floor(x_shifted/self.grid_resolution)
        return (i, j)


    def transform_grid_cell_to_point(self, grid_cell):
        '''Transforms the given grid cell (i, j) to a (x, y) point in the real   world.'''
        i, j= grid_cell
        x= j * self.grid_resolution - self.shift_x + self.grid_resolution/2
        y= i * self.grid_resolution - self.shift_y + self.grid_resolution/2
        return (x, y)    


    def is_cell_inside_map(self, cell):
        '''Gets a (row index, column index) cell and checks if it is inside the Occupancy Grid Map.'''
        row, column = cell
        is_inside= False
        if( (0 < row < self.map_height) and (0 < column < self.map_width) ):
            is_inside= True
        return is_inside

    
    def transform_particles_to_pose(self):
        '''Transform the list of particles to a list of particles, were each particle
        is a geometry_msg Pose.'''
        geometry_particles= []
        for particle in self.particles:
            x, y, theta= particle
            ang_x, ang_y, ang_z, w= quaternion_from_euler(0.0, 0.0, theta)
            position= Point(x= x, y=y, z= 0.0)
            orientation= Quaternion(x= ang_x, y= ang_y, z= ang_z, w= w)
            geometry_particles.append(Pose(position, orientation))
        return geometry_particles


    #____________________________________________________________________________________________________________________
    # Update 
    #____________________________________________________________________________________________________________________

    def update_particles(self, control, measurements):
        '''Update step of the particle Filter. Predicts the current pose of the robot based on the given
        (distance left wheel, distance right wheel) control. Then corrcts the predicted pose based on 
        the given (range, bearing) measurements.'''
        self.predict(control)     
        # self.correct(measurements) 
        

    #____________________________________________________________________________________________________________________
    # Test Algorithms (only for testing)
    #____________________________________________________________________________________________________________________

    def test_find_occupied_cells(self):
        # Find occupied cells
        occupied_cells= self.find_occupied_cells()
        # Check if cells are really occupied
        for occ_cell in occupied_cells:
            i, j= occ_cell[1]
            if(self.occupancy_grid_map[i][j] == self.occ):
                rospy.loginfo("is occupied")
            else:
                rospy.loginfo("is not occupied")
                pass
    

    def test_compute_likelihood_field(self):
        # self.compute_likelihood_field()
        self.compute_likelihood_field_with_NearestNeighbors()
        
#__________________________________________________________________________________________________________________________________
# Monte Carlo Localization  
#__________________________________________________________________________________________________________________________________


class MonteCarloLocalization():
    def __init__(self, update_rate, scan_topic, occupancy_grid_map, map_parameter, wheel_encoder_parameter, particle_filter_parameter):
        # Occupancy grid map data 
        # self.occupancy_grid_map= occupancy_grid_map
        # Map parameter
        start_pose, self.map_frame_id, self.map_width, self.map_height, self.origin_x, self.origin_y, self.grid_resolution, occupancy_values= map_parameter
        self.map_width_m= self.map_width * self.grid_resolution
        self.map_height_m= self.map_height * self.grid_resolution
        # Wheel Encoder data
        self.distance_left_wheel= 0
        self.distance_right_wheel= 0
        # Movement Threshold 
        control_threshold= 0.01
        self.control_threshold_squared= control_threshold**2
        # Wheel Encoder parameter
        wheel_encoder_topic, (self.wheel_encoder_motion_error_factor, self.wheel_encoder_turn_error_factor)= wheel_encoder_parameter
        # Lock object for save callback threads
        self.lock= threading.Lock()
        # Subscriber
        self.wheel_encoder= rospy.Subscriber(wheel_encoder_topic, WheelEncoder, self.wheel_encoder_callback)
        self.laser_scan_subscriber= rospy.Subscriber(scan_topic, LaserScan, self.laser_scan_callback)
        # Publisher 
        self.update_rate= update_rate
        self.particle_publisher= rospy.Publisher("particle_cloud", PoseArray, queue_size=5)
        # Init particle Filter
        map_parameter_particle_filter= (self.map_width_m, self.map_height_m, self.grid_resolution)
        self.particle_filter= ParticleFilter(particle_filter_parameter, start_pose, map_parameter_particle_filter, occupancy_values, occupancy_grid_map)
        # Measurements
        self.laser_scan= None
        # Test
        # self.occupancy_grid_object= OccupancyGrid() 
        # self.occupancy_grid_object.header.frame_id= "map"
        # self.init_occupancy_grid_message(map_width, map_height, origin_x, origin_y, grid_resolution)
        # # Test Publisher
        # self.map_publisher= rospy.Publisher("map", OccupancyGrid, queue_size=1)
        self.weight_publisher= rospy.Publisher("weights", Float64Array, queue_size=10)
        self.every_nth_ray= 2

    #____________________________________________________________________________________________________________________
    # Callback Functions
    #____________________________________________________________________________________________________________________
    def wheel_encoder_callback(self, distance):
        '''Accumulate the distances of the left and right wheel.'''
        self.lock.acquire()
        self.distance_left_wheel+= distance.left
        self.distance_right_wheel+= distance.right
        self.lock.release()


    def laser_scan_callback(self, laser_scan):
        '''Receive laser scan from topic.'''
        self.lock.acquire()
        self.laser_scan= laser_scan
        self.lock.release()

    #____________________________________________________________________________________________________________________
    # Publisher Functions
    #____________________________________________________________________________________________________________________
    
    def publish_particles(self):
        geomerty_poses= self.particle_filter.transform_particles_to_pose()
        particle_cloud= PoseArray()
        particle_cloud.header.stamp= rospy.Time.now()
        particle_cloud.header.frame_id= self.map_frame_id
        particle_cloud.poses= geomerty_poses
        self.particle_publisher.publish(particle_cloud)

    #____________________________________________________________________________________________________________________
    # Main methods
    #____________________________________________________________________________________________________________________

    def simulate_motion_error(self, left_wheel, right_wheel):
        '''Simulates gaussian error in robot motion. The error is simulated by two factors. 
        One is the error in distance the other is the error is due to slip while turning.'''
        left_distance, right_distance= (left_wheel, right_wheel)
        control_difference= left_distance - right_distance
        # Calculate error standarddeviation
        control_turn_variance= (self.wheel_encoder_turn_error_factor * control_difference)**2
        left_control_variance= (self.wheel_encoder_motion_error_factor * left_distance)**2 + control_turn_variance
        right_control_variance= (self.wheel_encoder_motion_error_factor * right_distance)**2 + control_turn_variance
        left_encoder_stddv= sqrt(left_control_variance)
        right_encoder_stddv= sqrt(right_control_variance)
        # Calculate distances with gaussian error. 
        left_distance_with_error= random.gauss(left_distance, left_encoder_stddv)
        right_distance_with_error=random.gauss(right_distance, right_encoder_stddv)
        return (left_distance_with_error, right_distance_with_error)


    @staticmethod
    def transform_laser_scan_to_measurement(laser_scan):
        '''Tranforms the sensor msgs LaserScan to a list of measurement's consisting of 
        (range, bearing) tuple.'''
        min_angle= laser_scan.angle_min
        angle_increment= laser_scan.angle_increment
        bearing= min_angle
        measurements= []
        # Transform LaserScan data
        for range in laser_scan.ranges:
            measurement= (range, bearing)
            bearing+= angle_increment
            measurements.append(measurement)
        return measurements    


    def transform_laser_scan_to_measurement_v2(self, laser_scan):
        '''Tranforms the sensor msgs LaserScan to a list of measurement's consisting of 
        (range, bearing) tuple. Only every nth measurement will be taken into account.'''
        min_angle= laser_scan.angle_min
        angle_increment= laser_scan.angle_increment
        bearing= min_angle
        measurements= []
        counter= 0
        # Transform LaserScan data
        for i in range(len(laser_scan.ranges)):
            # Only use every nth measurement
            if(not (counter % self.every_nth_ray)):
                r= laser_scan.ranges[i]
                measurements.append((r, bearing))
            bearing+= angle_increment
            counter+= 1
        return measurements

    
    def is_localization_necessary(self, control):
        '''Determines if localization is necessary based on the motion of the robot.'''
        is_necessary= False
        left_distance, right_distance= control
        if(left_distance**2 > self.control_threshold_squared and right_distance**2 > self.control_threshold_squared):
            is_necessary= True
        else: 
            is_necessary= False
        return is_necessary


    def execute(self):
        update_rate= rospy.Rate(self.update_rate)
        while not rospy.is_shutdown():
            # Check if Localization is necessary 
            if(self.is_localization_necessary((self.distance_left_wheel, self.distance_right_wheel))):
                self.lock.acquire()
                # Extract wheel encoder data
                distance_left_wheel= self.distance_left_wheel
                distance_right_wheel= self.distance_right_wheel
                self.distance_left_wheel= 0.0
                self.distance_right_wheel= 0.0
                # Get current measurements
                measurements= self.transform_laser_scan_to_measurement_v2(self.laser_scan)
                self.lock.release()
                # Simulate motion error
                control= self.simulate_motion_error(distance_left_wheel, distance_right_wheel)
                # Update pose by particle filter
                self.particle_filter.update_particles(control, measurements)
            else: 
                rospy.loginfo("\nNo Localization\n")
            # Publish particles
            self.publish_particles()
            self.publish_weights()
            update_rate.sleep()


    #________________________________________________________________________________________________________________
    # Test Part
    #________________________________________________________________________________________________________________


    # def init_occupancy_grid_message(self, map_width, map_height, origin_x, origin_y, grid_resolution):
    #     '''Init the static values of the OccupancyGrid message.'''
    #     self.occupancy_grid_object.info.width= map_width
    #     self.occupancy_grid_object.info.height= map_height
    #     self.occupancy_grid_object.info.origin.position.x= origin_x
    #     self.occupancy_grid_object.info.origin.position.y= origin_y
    #     self.occupancy_grid_object.info.resolution= grid_resolution 


    # def publish_map(self):
    #     self.occupancy_grid_object.data= self.occupancy_grid_map.ravel()
    #     self.map_publisher.publish(self.occupancy_grid_object)

    def publish_weights(self):
        weights= self.particle_filter.weights
        if(weights):
            weights_message= Float64Array()
            weights_message.data= weights
            self.weight_publisher.publish(weights_message)
            # rospy.loginfo("weights ")

    def test_find_occupied_cells(self):
        self.particle_filter.test_find_occupied_cells()


    def test_compute_likelihood_field_(self):
        self.particle_filter.test_compute_likelihood_field()


#__________________________________________________________________________________________________________________________________
#  Before the Algorithm
#__________________________________________________________________________________________________________________________________


def get_occupancy_grid_map(map_service_name="static_map", service_class= GetMap):
    '''Calling the service and receives the map. Extracts the map data.'''
    occupancy_grid_map= None
    rospy.wait_for_service("static_map")
    try:
        map_loader= rospy.ServiceProxy("static_map", service_class)
        occupancy_grid_map= map_loader()
        return occupancy_grid_map.map
    except rospy.ServiceException() as e:
        rospy.loginfo("The Service %s failed", e)


def extract_map_meta_data(occupancy_grid_map):
    '''Returns the parameter of the occupancy grid map given the occupancy_grid_map Message object.'''
    frame_id= occupancy_grid_map.header.frame_id
    map_width= occupancy_grid_map.info.width
    map_height= occupancy_grid_map.info.height
    origin_x= occupancy_grid_map.info.origin.position.x
    origin_y= occupancy_grid_map.info.origin.position.y
    grid_resolution= occupancy_grid_map.info.resolution
    return (frame_id, map_width, map_height, origin_x, origin_y, grid_resolution)


def transform_2D_grid_to_1D_grid(self, indice):
    '''Transforms a given 2D grid cell indice to an 1D grid cell index.'''
    row, column= indice
    index= row * self.number_of_grids_x + column
    return int(index)


#__________________________________________________________________________________________________________________________________
# Main
#__________________________________________________________________________________________________________________________________


def main():
    rospy.init_node("monte_carlo_localization_with_likelihood_field", anonymous=True)
    # Get the occupancy grid map from the ros map_server
    map_service_name="static_map"
    service_class= GetMap
    occupancy_grid_map_msg= get_occupancy_grid_map(map_service_name=map_service_name, service_class= service_class)
    # Extract map meta data from message
    frame_id, map_width, map_height, origin_x, origin_y, grid_resolution= extract_map_meta_data(occupancy_grid_map_msg)
    # Transform 1D map to 2D map
    occupancy_grid_map_2D= np.reshape(occupancy_grid_map_msg.data, (map_height, map_width))
    # Define start pose of robot 
    start_pose= (0.0, 0.0, 0.0)
    # Occupancy values
    occ= 100
    free= 0
    unknown= -1
    occupancy_values= (occ, free, unknown)
    # Summarize map parameter
    map_parameter= (start_pose, frame_id, map_width, map_height, origin_x, origin_y, grid_resolution, occupancy_values)
    # Robot chassis parameter (need to be received from .yaml later)
    h_chassis= 0.15
    dist_chassis_to_ground= h_chassis/5
    r_wheel= h_chassis/2 + dist_chassis_to_ground
    w_wheel= 0.3 * r_wheel
    r_chassis= 0.25
    wheel_separation= 2 * r_chassis + w_wheel
    # Motion Uncertainty 
    control_motion_factor = 0.35                    # Motion error in distance -> 35% error in distance
    control_turn_factor = 0.6                       # Motion error while turning, due to slip -> 60% error while turning.
    # Measurement Uncertainty
    measurement_distance_stddev = 0.2               # Distance measurement error [m].
    measurement_angle_stddev = 15.0 / 180.0 * pi    # Angle measurement error [rad].
    # Measurement parameter
    min_sensor_range= 0.1                           # Min measurement range [m]
    max_sensor_range= 8.0                           # Max measurement range [m]
    measurement_parameter= (min_sensor_range, max_sensor_range)
    # Summarize particle filter parameter
    uncertainty_parameter= (control_motion_factor, control_turn_factor, measurement_distance_stddev, measurement_angle_stddev)
    robot_parameter= (wheel_separation)
    particle_filter_parameter= (robot_parameter, uncertainty_parameter, measurement_parameter)
    # Wheel encoder information (for simulating wheel encoder data)
    wheel_encoder_topic= "wheel_encoder"
    wheel_encoder_motion_error_factor= 0.05          # 5% error in distance
    wheel_encoder_turn_error_factor= 0.15            # 15% error while turning 
    encoder_error= (wheel_encoder_motion_error_factor, wheel_encoder_turn_error_factor)
    wheel_encoder_parameter= (wheel_encoder_topic, encoder_error)
    # Publish

    # Subscriber
    scan_topic= "scan"
    
    # Define update rate
    update_rate= 4
    # Init Monte Carlo Localization
    mcl= MonteCarloLocalization(update_rate= update_rate, scan_topic= scan_topic, occupancy_grid_map= occupancy_grid_map_2D, 
                                map_parameter= map_parameter, wheel_encoder_parameter= wheel_encoder_parameter, 
                                particle_filter_parameter= particle_filter_parameter)
    # Execute algorithm
    # mcl.execute()
    start_time= rospy.get_time()
    mcl.test_compute_likelihood_field_()
    end_time= rospy.get_time()
    duration= end_time - start_time
    rospy.loginfo("Time for likelihood pre computation: %f", duration)
    rospy.loginfo("\n\n\nLikelihood Map Calculated\n")
    rospy.loginfo("\n\n\nLikelihood Map Calculated\n")
    rospy.loginfo("\n\n\nLikelihood Map Calculated\n")


if __name__=="__main__":
    main()
