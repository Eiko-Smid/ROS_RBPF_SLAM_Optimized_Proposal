#!/usr/bin/env python3

import rospy
# Python libarys
import threading
import numpy as np
from math import atan2, sin, cos, pi, sqrt, radians, degrees, isfinite, floor, exp 
import random
from scipy.stats import norm as normal_dist
# Messages and Services
from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion
from sensor_msgs.msg import LaserScan
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import tf
# Custom messages 
from rvc_commander.msg import WheelEncoder
from rvc_commander.msg import Float64Array


'''
Description:

    Monte Carlo Localization Algorithm with uses the mixture beam based model based on the probabilistic robotics book. 
    
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
        self.control_motion_factor, self.control_turn_factor, measurement_model_parameter= uncertainty_parameter
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
        # self.initialize_particles_v2(start_pose, map_width_m, map_height_m)
        # Robot parameter.
        self.wheel_saperation= wheel_saperation
        # Measurement parameter
        self.min_sensor_range, self.max_sensor_range= measurement_parameter
        # Measurement model parameter
        self.sigma_hit, self.z_hit, self.z_short, self.z_max, self.z_random, self.lambda_short= measurement_model_parameter
        # Resampling 
        number_of_particles= len(self.particles)
        self.neff_threshold= number_of_particles * (1/2)
        self.weights= []
        self.initialize_weights()
        # Test variables____________________________________________________________________________
        self.flag= True
        self.normalized_weights= None
        

    def initialize_particles(self, start_pose, map_width_m, map_height_m):
        '''Gets the known start pose of the robot and initialize the particles 
        around the given pose.'''
        self.particles= []
        standard_deviation= (0.5, 0.5, 10*pi/180)
        number_of_particles= 70
        
        for i in range(number_of_particles):
            particle= [ random.gauss(start_pose[j], standard_deviation[j]) for j in range(3)]
            self.particles.append(particle)


    def initialize_particles_v2(self, start_pose, map_width_m, map_height_m):
        '''Gets the known start pose of the robot and initialize the particles 
        around the given pose.'''
        self.particles= []
        pose= [0.0, 0.0, 0.0]
        number_of_particles= 70
        
        for i in range(number_of_particles):
            particle = pose.copy()
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

    def exclude_invalid_measurements(self, measurements):
        '''Gets s list of (range, bearing) measurements. Determins which measurements are valid. Returns a list of all 
        valid measurements.'''
        valid_measurements= []
        
        for m in measurements:
            range, bearing= m
            if(range >= self.min_sensor_range and range <= self.max_sensor_range and isfinite(range)):
                valid_measurements.append(m)
        
        return valid_measurements


    def find_reflecting_grid_cell(self, measurement, pose):
        '''Gets a (range, bearing) measurement and a (x, y, heading) pose and calculates 
        the indices of the reflecting grid cell. Also checks if the measurement range is 
        in the area of the sensor range and if the range is infinite. If there is no 
        plausible measurement, then the function return None, otherwise the reflected 
        grid cell.'''
        x, y, heading= pose
        range, bearing= measurement
        reflecting_cell= ()
        
        # Calculate x,y-position of reflected beam.
        phi= atan2(sin(heading + bearing), cos(heading + bearing))
        reflection_point_x= x + range * cos(phi)
        reflection_point_y= y + range* sin(phi)
        
        # Transfrom cell coordinates to cell indices.
        reflecting_cell= self.transform_point_to_grid_cell((reflection_point_x, reflection_point_y))
        
        return reflecting_cell


    def find_occupied_grid_cell(self, start_grid_idx, end_grid_idx):
        '''Finds occupied the occupied cell between the start grid and the 
        end grid. '''
        #  y= lines, x = column 
        y_start, x_start= start_grid_idx
        y_end, x_end= end_grid_idx
        affected_cells= []
        
        # Determine if slope is rising or falling -> y increment up or down
        dx= x_end - x_start
        dy= y_end - y_start
        
        # Define Increments 
        increment_x= np.sign(dx)
        increment_y= np.sign(dy)
        if(dx<0): dx= -dx
        if(dy<0): dy= -dy
        
        # Set parameters
        ddx= increment_x
        ddy= increment_y
        if(dx > dy):
            pdx= increment_x
            pdy= 0
            slow_direction= dy
            fast_direction= dx
        else:
            pdx= 0 
            pdy= increment_y
            slow_direction= dx
            fast_direction= dy
        
        # Initialization
        x= x_start
        y= y_start
        err= fast_direction / 2.0
        
        # Indicator if occupied cell was found (only in mode 2)
        is_occupied= False
        is_inside= False
        leaf_loop= False
        index= 0
        affected_cells.append(start_grid_idx)
        
        # Algorithm
        while((index < fast_direction) and (not leaf_loop)):
            index+= 1
            err-= slow_direction
            
            if(err < 0):
                err+= fast_direction
                x+= ddx
                y+= ddy
            else:
                x+= pdx
                y+= pdy
            
            # Add new cell to list
            affected_cells.append((y, x))  
            
            if(self.is_cell_inside_map((y, x))):
                is_inside= True
                # Check if cell is occupied
                occupancy_value= self.occupancy_grid_map[y][x]
                if(occupancy_value == self.occ):
                    leaf_loop= True
                    is_occupied= True
            else: 
                is_inside= False
                leaf_loop= True
        
        return affected_cells, (y, x), is_occupied, is_inside
                
    
    def predict_measurement(self, particle, particle_cell_index, measurement):
        '''Predict the measurement based on the given (x, y, theta) particle pose and the (range, bearing) measurement. 
        The given particle must be inside the map. The measurement needs to be a valid measurement (not inf for example and 
        in range parameter of sensor). Returns the predicted range between the particle pose and the grid cell, which should 
        have reflected the beam, if the particle is the true pose of the robot. 
        If the algorithm detects a cell which occupancy value is "unknown" or if it detects a cell which is outside the map,
        these cell will be treated as if it's occupancy value is "occupied".'''
        px, py, theta= particle
        range, bearing= measurement
        
        # Compute grid cell indices of max measurement
        max_measurement_grid_cell= self.find_reflecting_grid_cell((self.max_sensor_range, bearing), particle)
        
        # Estimate reflecting grid cell 
        affected_cells, last_cell, is_occupied, is_inside= self.find_occupied_grid_cell(particle_cell_index, max_measurement_grid_cell)
        
        # Is cell occupied or unknown or out of map or max range, treat the cell as if the cell is occupied and predict the measurement
        rx, ry= self.transform_grid_cell_to_point(last_cell)
        
        # Calculate range and bearing of reflecting cell
        dx= rx - px
        dy= ry - py
        predicted_range= sqrt(dx**2 + dy**2)
        return predicted_range, is_inside, is_occupied


    def calculate_measurement_probability(self, measurement, predicted_measurement):
        """Given a measurement and a predicted measurement, computes the probability that the predicted 
        measurement fit's to the measurement."""
        measured_range, measured_bearing= measurement
        predicted_range= predicted_measurement
        range_difference= measured_range - predicted_range
        
        # Normal measurement probability (without normalization)
        probability= self.z_hit * normal_dist.pdf(range_difference, 0, self.sigma_hit)
        
        # Probability for unexpected objects (without normalization)
        if(range_difference < 0):
            probability+= self.z_short * self.lambda_short * exp((-self.lambda_short) * measured_range)
        
        # Take measurement failures into account
        if(measured_range == self.max_sensor_range):
            probability+= self.z_max * 1.0
            rospy.loginfo("\nMax Range reading\n")
        
        # Take random measurements into account
        if(measured_range < self.max_sensor_range):
            probability+= self.z_random * 1/self.max_sensor_range
        
        return probability


    def compute_weights(self, measurements):
        '''Compute a weight for each particle depending on the measurements.'''
        index= 0
        # Extract all valid measurements
        valid_measurements= self.exclude_invalid_measurements(measurements)
        rospy.loginfo("Number of valid measurements %i and measurements %i", len(valid_measurements), len(measurements))
        # Check if there are valid measurements
        if(valid_measurements):
            for particle in self.particles:  
                # Check if particle is inside the map if not -> weight= 0
                particle_weight= 1.0
                weight= 1.0
                x, y, theta= particle
                particle_cell_index= self.transform_point_to_grid_cell((x, y))
                if(self.is_cell_inside_map(particle_cell_index)):
                    for m in valid_measurements:                         
                        # Predict the measurement
                        predicted_measurement, is_inside, is_occupied= self.predict_measurement(particle, particle_cell_index, m)                        
                        # Compute measurement probability
                        weight*= self.calculate_measurement_probability(m, predicted_measurement)
                    # Update particle weight
                    particle_weight*= weight
                else:
                    # Make sure particle outside map will be erased
                    particle_weight= 0.0    
                self.weights[index]= self.weights[index] * particle_weight
                index+= 1
            # Normalize all weights
            normalizer= sum(self.weights)
            self.weights= [w / normalizer for w in self.weights] 
        else:
            rospy.loginfo("No Valid Measurements")


    def compute_weights_test(self, measurements):
        '''Compute a weight for each particle depending on the measurements. AMCL Variant!
        The weights will not be multiplied, we use an ad-hoc-scheme.'''
        index= 0
        # Extract all valid measurements
        valid_measurements= self.exclude_invalid_measurements(measurements)
        rospy.loginfo("Number of valid measurements %i and measurements %i", len(valid_measurements), len(measurements))
        # Check if there are valid measurements
        if(valid_measurements):
            for particle in self.particles:  
                # Check if particle is inside the map if not -> weight= 0
                particle_weight= 1.0
                measurement_weight= 1.0
                x, y, theta= particle
                particle_cell_index= self.transform_point_to_grid_cell((x, y))
                if(self.is_cell_inside_map(particle_cell_index)):
                    for m in valid_measurements:                         
                        # Predict the measurement
                        predicted_measurement, is_inside, is_occupied= self.predict_measurement(particle, particle_cell_index, m)                        
                        # Compute measurement probability
                        w= self.calculate_measurement_probability(m, predicted_measurement)
                        measurement_weight+= w * w * w
                    # Update particle measurement_weight
                    particle_weight*= measurement_weight
                else:
                    # Make sure particle outside map will be erased
                    particle_weight= 0.0    
                self.weights[index]= self.weights[index] * particle_weight
                index+= 1
            # Normalize all weights
            normalizer= sum(self.weights)
            self.weights= [w / normalizer for w in self.weights] 
        else:
            rospy.loginfo("No Valid Measurements")


    def initialize_weights(self):
        '''Initialize the weights where each weight is 1/number_of_weights.''' 
        number_of_weights= len(self.particles)
        weight= 1.0/number_of_weights
        self.weights= [weight for i in range(number_of_weights)]


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
        '''Correction step of the particle filter.'''
        # First compute all weights.
        self.compute_weights_test(measurements)
        
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
        x_shifted = x + self.shift_x
        y_shifted = y + self.shift_y
        i = floor(y_shifted/self.grid_resolution)
        j = floor(x_shifted/self.grid_resolution)
        return (i, j)


    def transform_grid_cell_to_point(self, grid_cell):
        '''Transforms the given grid cell (i, j) to a (x, y) point in the real world.'''
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

    
    def extract_density(self):
        '''
        Extract the density of the particles. The density is estimated by calculating the mean position and the 
        mean heading vector of all particles. The mean heading vector is calculated by summing up all heading 
        vectors of the particles and then calculating the angle of the resulting vector. The mean position is 
        calculated by summing up all x and y values of the particles and then dividing by the number of particles.
        ''' 
        sum_x= 0.0
        sum_y= 0.0
        theta_x= 0.0
        theta_y= 0.0
        number_of_particles= 0
        
        # Compute mean position and mean heading vector for each particle
        for particle in self.particles:
            x, y, theta= particle
            
            # Mean position
            sum_x+= x
            sum_y+= y
            
            # Mean heading vector
            theta_x+= cos(theta)
            theta_y+= sin(theta)
            number_of_particles+= 1
        
        return (sum_x/number_of_particles, sum_y/number_of_particles, atan2(theta_y, theta_x))  



    #____________________________________________________________________________________________________________________
    # Update 
    #____________________________________________________________________________________________________________________

    def update_particles(self, control, measurements):
        '''Update step of the particle Filter. Predicts the current pose of the robot based on the given
        (distance left wheel, distance right wheel) control. Then corrcts the predicted pose based on 
        the given (range, bearing) measurements.'''
        self.predict(control)     
        time_before_correction= rospy.get_time()
        self.correct(measurements) 
        time_after_correction= rospy.get_time()
        rospy.loginfo("\ntime for correction %f [s]\n", (time_after_correction-time_before_correction))
        

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
        
#__________________________________________________________________________________________________________________________________
# Monte Carlo Localization  
#__________________________________________________________________________________________________________________________________


class MonteCarloLocalization():
    '''Class for communication with ROS.'''
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
        # TF: odom -> base_link is expected from odometry / simulator, this node publishes map -> odom
        self.tf_listener = tf.TransformListener()
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.odom_frame_id = rospy.get_param("~odom_frame_id", "odom_link")
        self.base_frame_id = rospy.get_param("~base_frame_id", "base_link")
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
        self.pose_publisher= rospy.Publisher("pose", Pose, queue_size=1)
        # Only use every n'th particle.
        self.every_nth_ray= 20

    #____________________________________________________________________________________________________________________
    # Callback Methods
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
    # Publisher Methods
    #____________________________________________________________________________________________________________________
    
    def publish_particles(self):
        '''Publishes the particles to the given topic.'''
        geomerty_poses= self.particle_filter.transform_particles_to_pose()
        particle_cloud= PoseArray()
        particle_cloud.header.stamp= rospy.Time.now()
        particle_cloud.header.frame_id= self.map_frame_id
        particle_cloud.poses= geomerty_poses
        self.particle_publisher.publish(particle_cloud)


    def publish_pose(self, pose):
        # Extract pose
        x, y, theta = pose
        pose= Pose()
        pose.position.x= x
        pose.position.y= y
        
        # Transform angles to quaternion
        ang_x, ang_y, ang_z, w= quaternion_from_euler(0.0, 0.0, theta)
        orientation= Quaternion(x= ang_x, y= ang_y, z= ang_z, w= w)
        pose.orientation= orientation
        self.pose_publisher.publish(pose)


    @staticmethod
    def normalize_angle(angle):
        return atan2(sin(angle), cos(angle))


    def publish_map_to_odom_tf(self, pose):
        """
        This function computes the transformation between the odom frame and the map frame based on the tf between
        the odom frame and the base frame as well as the pose of the robot. The pose of the robot is the tf between 
        the map frame and the base frame, which is estimated by the MCL algorithm. 
        
        """
        try:
            # get tf: odom -> base_link
            (translation, rotation) = self.tf_listener.lookupTransform(
                self.odom_frame_id, self.base_frame_id, rospy.Time(0)
            )
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn_throttle(2.0, "MCL: could not lookup TF %s -> %s", self.odom_frame_id, self.base_frame_id)
            return

        odom_x = translation[0]
        odom_y = translation[1]
        (_, _, odom_theta) = euler_from_quaternion(rotation)

        # Particle filter estimate: map -> base_link
        pose_x, pose_y, pose_theta = pose
        rospy.loginfo(f"\nPose used for TF is: {pose}")

        # Compute map -> odom from map -> base_link and odom -> base_link
        map_to_odom_theta = self.normalize_angle(pose_theta - odom_theta)
        c = cos(map_to_odom_theta)
        s = sin(map_to_odom_theta)

        map_to_odom_x = pose_x - (c * odom_x - s * odom_y)
        map_to_odom_y = pose_y - (s * odom_x + c * odom_y)

        q = quaternion_from_euler(0.0, 0.0, map_to_odom_theta)
        self.tf_broadcaster.sendTransform(
            (map_to_odom_x, map_to_odom_y, 0.0),
            q,
            rospy.Time.now(),
            self.odom_frame_id,
            self.map_frame_id
        )

    #____________________________________________________________________________________________________________________
    # Main methods
    #____________________________________________________________________________________________________________________

    def simulate_motion_error(self, left_wheel, right_wheel):
        '''Simulates gaussian error in robot motion. The error is simulated by two factors. 
        One is the error in distance the other is the error due to slip while turning.'''
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


    def transform_laser_scan_to_measurement(self, laser_scan):
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
        if(left_distance**2 > self.control_threshold_squared or right_distance**2 > self.control_threshold_squared):
            is_necessary= True
        else: 
            is_necessary= False
        return is_necessary


    def execute(self):
        update_rate= rospy.Rate(self.update_rate)
        
        # Wait for tf to be available
        rospy.loginfo("Waiting for TF odom -> base_link...")
        self.tf_listener.waitForTransform(
            self.odom_frame_id,
            self.base_frame_id,
            rospy.Time(0),
            rospy.Duration(5.0)
        )
        rospy.loginfo("TF available!")

        while not rospy.is_shutdown():
            # Check if measurement is available, else stop rest of loop
            if self.laser_scan is None:
                    rospy.logwarn_throttle(2.0, "Waiting for laser scan measurement initialization...")
                    update_rate.sleep()
                    continue
        
            # Check if Localization is necessary 
            if(self.is_localization_necessary((self.distance_left_wheel, self.distance_right_wheel))):
                self.lock.acquire()
                # Extract wheel encoder data
                distance_left_wheel= self.distance_left_wheel
                distance_right_wheel= self.distance_right_wheel
                self.distance_left_wheel= 0.0
                self.distance_right_wheel= 0.0
                # Extract laser scan measurements
                laser_scan = self.laser_scan
                self.lock.release()
                # Get current measurements
                measurements= self.transform_laser_scan_to_measurement(laser_scan)
                # Simulate motion error
                control= self.simulate_motion_error(distance_left_wheel, distance_right_wheel)
                # Update pose by particle filter
                self.particle_filter.update_particles(control, measurements)
            else: 
                rospy.loginfo("\nNo Localization\n")
            
            # Extract density -> pose
            pose = self.particle_filter.extract_density()

            # Publish particles
            self.publish_particles()

            # Publish weights
            self.publish_weights()
            
            # Publish pose
            self.publish_pose(pose=pose)

            # Publish correction TF: map -> odom
            self.publish_map_to_odom_tf(pose=pose)
            
            update_rate.sleep()


    #________________________________________________________________________________________________________________
    # Test Part
    #________________________________________________________________________________________________________________

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
    rospy.init_node("monte_carlo_localization_with_beam_model", anonymous=True)
    # Get the occupancy grid map from the ros map_server
    map_service_name="static_map"
    service_class= GetMap
    occupancy_grid_map_msg= get_occupancy_grid_map(map_service_name=map_service_name, service_class= service_class)
    # Extract map meta data from message
    frame_id, map_width, map_height, origin_x, origin_y, grid_resolution= extract_map_meta_data(occupancy_grid_map_msg)
    # Transform 1D map to 2D map
    occupancy_grid_map_2D= np.reshape(occupancy_grid_map_msg.data, (map_height, map_width))
    rospy.loginfo(f"\n\nThe OGM includes the following unique values:\n{np.unique(occupancy_grid_map_2D)}\n\n")
    
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
    
    # Measurement parameter
    min_sensor_range= 0.1                           # Min measurement range [m]
    max_sensor_range= 8.0                           # Max measurement range [m]
    # [zhit zshort zmax zrand sigma_hit zmax_range lambda_short]
    # [0.40  0.20  0.25  0.15  0.1  4  .5]
    sigma_hit = 0.15                                # Distance measurement error [m].
    z_hit= 0.5                                      # z_hit parameter for measurement model
    z_short= 0.2
    z_max= 0.15
    z_random= 0.15                                  # z_rand parameter for measurement model
    lambda_short= 0.5
    # sigma_hit, z_hit, z_short, z_max, z_random, lambda_short= [0.40,  0.20,  0.25,  0.15,  0.1,  4.5]
    sigma_hit, z_hit, z_short, z_max, z_random, lambda_short= [0.15,  0.35,  0.2,  0.25,  0.15,  0.5]
    measurement_model_parameter= (sigma_hit, z_hit, z_short, z_max, z_random, lambda_short)
    measurement_parameter= (min_sensor_range, max_sensor_range)
    
    # Summarize particle filter parameter
    uncertainty_parameter= (control_motion_factor, control_turn_factor, measurement_model_parameter)
    robot_parameter= (wheel_separation)
    particle_filter_parameter= (robot_parameter, uncertainty_parameter, measurement_parameter)
    
    # Wheel encoder information (for simulating wheel encoder data)
    wheel_encoder_topic= "wheel_encoder"
    wheel_encoder_motion_error_factor= 0.05          # 5% error in distance
    wheel_encoder_turn_error_factor= 0.15            # 15% error while turning 
    encoder_error= (wheel_encoder_motion_error_factor, wheel_encoder_turn_error_factor)
    wheel_encoder_parameter= (wheel_encoder_topic, encoder_error)

    # Subscriber
    scan_topic= "scan"
    
    # Define update rate
    update_rate= 4
    
    # Init Monte Carlo Localization
    mcl= MonteCarloLocalization(update_rate= update_rate, scan_topic= scan_topic, occupancy_grid_map= occupancy_grid_map_2D, 
                                map_parameter= map_parameter, wheel_encoder_parameter= wheel_encoder_parameter, 
                                particle_filter_parameter= particle_filter_parameter)
    
    # Execute algorithm
    mcl.execute()



if __name__=="__main__":
    main()
