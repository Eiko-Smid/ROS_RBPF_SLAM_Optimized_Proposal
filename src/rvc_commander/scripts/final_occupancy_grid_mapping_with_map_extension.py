#!/usr/bin/env python3

import rospy
import threading
import numpy as np
from math import exp, atan2, sin, cos, radians, degrees, floor, ceil, isfinite
import time
from geometry_msgs.msg import Pose, Point
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
# from nav_msgs.msg import OccupancyGrid
from rvc_commander.msg import Measurement
from rvc_commander.msg import LogOddsMap


'''
The goal of this program is to see, if a change in the map effects the result of the grid map.


    1) Make a grid map were all cells are black
    2) Publish the map
    3) Extend the map 
    4) Publish the extended map and see the difference 

'''

    
#__________________________________________________________________________________________________________________________
# Occupancy Grid Mapping Class  
#__________________________________________________________________________________________________________________________


class OccupancyGridMapping():
    IDX_X= 0
    IDX_Y= 1
    def __init__(self, map_parameter, occupancy_parameter, sensor_parameter):
        '''Class for creating a grid map with the possibility. Arrays are numpy arrays. Attention! After creating
        the grid map with "create_map()" the map will stay in log Odds space forever. To get the occupancy grid map
        out of the self.log_odds_map, the two methodes "transform_log_odds_map_to_occupancy_grid_map" and 
        "transform_log_odds_map_to_probability_map" need to be used.'''     
        # Extract parameter
        self.map_width_m, self.map_height_m, self.grid_resolution_m, self.min_distance_to_border= map_parameter
        prior_probability, increasing_probability, decreasing_probability, self.min_log_odds, self.max_log_odds= occupancy_parameter
        self.min_sensor_range, self.max_sensor_range= sensor_parameter
        # Define map
        self.log_odds_map= []                                                          
        self.number_of_cells_x= 0.0
        self.number_of_cells_y= 0.0
        self.left_map_border_m= 0.0
        self.top_map_border_m= 0.0
        self.right_map_border_m= 0.0
        self.bottom_map_border_m= 0.0
        # Variables needed for point to grid cell transformation
        self.shift_x= 0
        self.shift_y= 0
        # Create OccupancyGrid Message object
        lom= LogOddsMap()
        self.log_odds_map_msg= LogOddsMap()
        self.log_odds_map_msg.header.frame_id= "log_odds_map"
        # Ensure correct prior probability
        if(prior_probability <= 0 or prior_probability > 1.0):                  
            self.log_odds_prior= np.log(0.5 / (1 - 0.5))    
            rospy.loginfo("\nTHe prior probability must lie between 0 and 1.\n")
            rospy.loginfo("The prior was set to: %f", 0.5)
        else:
            self.log_odds_prior= np.log(prior_probability/(1-prior_probability))    # Calculate log Odds of prior 
        # Ensure correct increasing probability
        if(increasing_probability <= 0 or increasing_probability > 1.0):
            self.log_odds_increasing_probability= np.log(0.65 / 0.35) 
            rospy.loginfo("\nThe increasing probability must lie between 0 and 1.\n")
            rospy.loginfo("The increasing probability was set to: %f", 0.65)
        else:
            self.log_odds_increasing_probability= np.log(increasing_probability / (1 - increasing_probability))
        # Ensure correct decreasing probability
        if(decreasing_probability <= 0 or decreasing_probability >= 1.0):
            self.log_odds_decreasing_probability= np.log(0.35 / 0.65)
            rospy.loginfo("\nThe decreasing probability must lie between 0 and 1.\n")
            rospy.loginfo("The decreasing probability was set to: %f", 0.35)
        else:
            self.log_odds_decreasing_probability= np.log(decreasing_probability / (1 - decreasing_probability))

    
    #_______________________________________________________________________________________________________________
    # Map creation
    #_______________________________________________________________________________________________________________

    def create_map(self):
        '''Create map and init prior probability'''
        # Define number of grids in x direction (must be odd value)
        self.number_of_cells_x= ceil(self.map_width_m / self.grid_resolution_m)        
        # Check for odds number of grid cells
        if(not (self.number_of_cells_x % 2)):
            self.number_of_cells_x+= 1
        # Update map width
        self.map_width_m= self.number_of_cells_x * self.grid_resolution_m
        # Define number of grids in y direction(must be odd value)
        self.number_of_cells_y= ceil(self.map_height_m / self.grid_resolution_m)        
        # Check for odds number of grid cells
        if(not (self.number_of_cells_y % 2)):
            self.number_of_cells_y+= 1
        # Update map Height
        self.map_height_m= self.number_of_cells_y * self.grid_resolution_m
        # Create map and initialize prior probability 
        self.log_odds_map= np.full((self.number_of_cells_y, self.number_of_cells_x), self.log_odds_prior)
        # Init variables needed transformation (point -> cell)
        self.shift_x= self.map_width_m / 2
        self.shift_y= self.map_height_m / 2
        # Init OccupancyGrid message
        self.init_occupancy_grid_message()
        # Define the border values for the map
        half_map_width= self.map_width_m / 2.0
        half_map_height= self.map_height_m / 2.0
        self.left_map_border_m= -half_map_width 
        self.top_map_border_m= half_map_height
        self.right_map_border_m= half_map_width
        self.bottom_map_border_m= - half_map_height        

    
    def init_occupancy_grid_message(self):
        '''Init the static values of the OccupancyGrid message.'''
        self.log_odds_map_msg.info.width= self.number_of_cells_x
        self.log_odds_map_msg.info.height= self.number_of_cells_y
        origin_x, origin_y= self.transform_grid_cell_to_point((0, 0))
        self.log_odds_map_msg.info.origin.position.x= origin_x
        self.log_odds_map_msg.info.origin.position.y= origin_y
        self.log_odds_map_msg.info.resolution= self.grid_resolution_m
    
    #_______________________________________________________________________________________________________________
    # Map access and map manipulation
    #_______________________________________________________________________________________________________________

    def return_log_odds_map(self):
        '''Retruns the grid map in log odds form.'''
        return self.log_odds_map

    
    def return_log_odds_map_object(self):
        # Copy the logOdds map to the message
        self.log_odds_map_msg.data= self.log_odds_map.ravel()
        # generate timestamp
        self.log_odds_map_msg.header.stamp= rospy.Time.now()        
        return self.log_odds_map_msg

    
    def extend_map(self, direction, distance):
        # Calculate number of cells to extend
        number_of_cells= ceil(distance / self.grid_resolution_m)     
        was_extension_successfull= True   
        old_log_odds_map= np.copy(self.log_odds_map)
        # Extend map on the left
        if(direction == "l"):
            self.number_of_cells_x+= number_of_cells       
            if(not (self.number_of_cells_x % 2)):
                self.number_of_cells_x+= 1   
                number_of_cells+= 1       
            # Update map parameter
            self.map_width_m= self.number_of_cells_x * self.grid_resolution_m 
            extended_distance= number_of_cells * self.grid_resolution_m
            self.shift_x+= extended_distance
            self.left_map_border_m-= extended_distance
            # Create new map 
            self.log_odds_map= np.full((self.number_of_cells_y, self.number_of_cells_x), self.log_odds_prior)
            # Copy old map to the right pose, of the new map
            self.log_odds_map[:, number_of_cells:]= old_log_odds_map    
        # Extend map on the right
        elif(direction == "r"):
            old_number_of_cells_x= self.number_of_cells_x
            self.number_of_cells_x+= number_of_cells
            if(not (self.number_of_cells_x % 2)):
                self.number_of_cells_x+= 1  
                number_of_cells+=1   
            # Update map parameter         
            self.map_width_m= self.number_of_cells_x * self.grid_resolution_m               
            self.right_map_border_m+= number_of_cells * self.grid_resolution_m
            # Create new map 
            self.log_odds_map= np.full((self.number_of_cells_y, self.number_of_cells_x), self.log_odds_prior)
            # Copy old map to the right pose, of the new map
            self.log_odds_map[:, :old_number_of_cells_x]= old_log_odds_map
        # Extend map on the bottom
        elif(direction == "b"):
            self.number_of_cells_y+= number_of_cells
            if(not (self.number_of_cells_y % 2)):
                self.number_of_cells_y+= 1  
                number_of_cells+= 1    
            # Update map parameter 
            self.map_height_m= self.number_of_cells_y * self.grid_resolution_m      
            extended_distance= number_of_cells * self.grid_resolution_m
            self.shift_y+= extended_distance
            self.bottom_map_border_m-= extended_distance
            # Create new map 
            self.log_odds_map= np.full((self.number_of_cells_y, self.number_of_cells_x), self.log_odds_prior)
            # Copy old map to the right pose, of the new map
            self.log_odds_map[number_of_cells:, :]= old_log_odds_map
        # Extend map on the top
        elif(direction == "t"):
            old_number_of_cells_y= self.number_of_cells_y
            self.number_of_cells_y+= number_of_cells
            if(not (self.number_of_cells_y % 2)):
                self.number_of_cells_y+= 1                
                number_of_cells+= 1
            # Update map parameter 
            self.map_height_m= self.number_of_cells_y * self.grid_resolution_m
            self.top_map_border_m+= number_of_cells * self.grid_resolution_m
            # Create new map 
            self.log_odds_map= np.full((self.number_of_cells_y, self.number_of_cells_x), self.log_odds_prior)
            # Copy old map to the right pose, of the new map
            self.log_odds_map[:old_number_of_cells_y, :]= old_log_odds_map
        else:
            was_extension_successfull= False
        return number_of_cells, was_extension_successfull


    def map_extension_if_necessary(self, pose):
        x, y, theta= pose        
        extension_needed= False
        # Check if map needed to be extended on the left side 
        if((x - self.min_distance_to_border) < self.left_map_border_m):
            # rospy.loginfo("position= %f, %f", x, y)
            # rospy.loginfo("")
            self.extend_map(direction='l', distance= self.min_distance_to_border)
            extension_needed= True
            rospy.loginfo("left side extended")
        # Check if map needed to be extended on the right side 
        elif((x + self.min_distance_to_border) > self.right_map_border_m):
            self.extend_map(direction='r', distance= self.min_distance_to_border)
            extension_needed= True
            rospy.loginfo("right side extended")
        # Check if map needed to be extended on the bottom side 
        if((y - self.min_distance_to_border) < self.bottom_map_border_m):
            self.extend_map(direction='b', distance= self.min_distance_to_border)
            extension_needed= True
            rospy.loginfo("bottom extended")
        # Check if map needed to be extended on the top side 
        elif((y + self.min_distance_to_border) > self.top_map_border_m):            
            self.extend_map(direction='t', distance= self.min_distance_to_border)
            extension_needed= True
            rospy.loginfo("top extended")
        if(extension_needed):
            self.update_log_odds_message()
        return extension_needed


    def update_log_odds_message(self):
        self.log_odds_map_msg.info.width= self.number_of_cells_x
        self.log_odds_map_msg.info.height= self.number_of_cells_y
        origin_x, origin_y= self.transform_grid_cell_to_point((0, 0))
        self.log_odds_map_msg.info.origin.position.x= origin_x
        self.log_odds_map_msg.info.origin.position.y= origin_y

    #_______________________________________________________________________________________________________________
    # Transformations
    #_______________________________________________________________________________________________________________

    @staticmethod
    def log_odds_to_probability(log_odds):
        '''Calculates the probability according to the given log Odds value.'''
        log_odds_exp= exp(log_odds)
        return log_odds_exp / (1+ log_odds_exp)


    @staticmethod
    def probability_to_occupancy(probability):
        occupancy_value= 0.0
        if(probability < 0.5):
            occupancy_value= 0.0
        elif(probability == 0.5):
            occupancy_value= 0.5
        else:
            occupancy_value= 1.0
        return occupancy_value


    @staticmethod
    def transform_log_odds_map_to_probability_map(log_odds_map):
        '''Transfers the map from the log Odds space to the 
        probability space and returns the transformened map.'''
        probability_map= np.copy(log_odds_map)
        map_shape= np.shape(probability_map)
        for i in range(map_shape[0]):
             for j in range(map_shape[1]):
                #  probability_map[i][j]= self.log_odds_to_probability(self.log_odds_map[i][j])
                probability_map[i][j]= OccupancyGridMapping.log_odds_to_probability(log_odds_map[i][j])
        return np.copy(probability_map)

    
    @staticmethod
    def transform_probability_map_to_occupancy_map(probability_grid_map):
        '''Transforms the given grid map from probability space to occupancy space.'''
        occupancy_grid_map= np.copy(probability_grid_map)
        map_shape= np.shape(probability_grid_map)
        # Round probability of each grid cell to 0, 1, or 0.5
        for i in range(map_shape[0]):
            for j in range(map_shape[1]):
                occupancy_grid_map[i][j]= OccupancyGridMapping.probability_to_occupancy(probability_grid_map[i][j])
        return occupancy_grid_map

    
    @staticmethod
    def transform_log_odds_map_to_occupancy_grid_map(log_odds_map):
        '''Transforms the given map from log odds space to occupancy space.'''
        probability_grid_map= OccupancyGridMapping.transform_log_odds_map_to_probability_map(log_odds_map)
        occupancy_grid_map= OccupancyGridMapping.transform_probability_map_to_occupancy_map(probability_grid_map)
        return occupancy_grid_map


    def transform_point_to_grid_cell(self, point):
        '''Transforms an (x, y) point to the array access indices (i, j for row, column). '''
        x,y = point
        x_shifted= x + self.shift_x
        y_shifted= y + self.shift_y
        i= floor(y_shifted/self.grid_resolution_m)
        j= floor(x_shifted/self.grid_resolution_m)
        return (i, j)


    def transform_grid_cell_to_point(self, grid_cell):
        '''Transforms the given grid cell (i, j) to a (x, y) point in the real world.'''
        i, j= grid_cell
        x= j * self.grid_resolution_m - self.shift_x + self.grid_resolution_m/2
        y= i * self.grid_resolution_m - self.shift_y + self.grid_resolution_m/2
        return (x, y)    



    #_______________________________________________________________________________________________________________
    # Main Algorithm
    #_______________________________________________________________________________________________________________

    def find_reflecting_grid_cell(self, measurement, pose):
        '''Gets a (range, bearing) measurement and a (x, y, heading) pose and calculates 
        the indices of the reflecting grid cell. Also checks if the measurement range is 
        in the area of the sensor range and if the range is infinite. If there is no 
        plausible measurement, then the function return None, otherwise the reflected 
        grid cell.'''
        x, y, heading= pose
        range, bearing= measurement
        reflecting_cell= ()
        # Check if range is in max sensor range or infinite
        if(range <= self.min_sensor_range or range >= self.max_sensor_range or not isfinite(range)) :
            # There is no reflecting cell
            reflecting_cell= None
        else: 
            # Ensure angles between -pi and pi
            bearing= atan2(sin(bearing), cos(bearing))
            heading= atan2(sin(heading), cos(heading))
            # Calculate x,y-position of reflected beam.
            phi= atan2(sin(heading + bearing), cos(heading + bearing))
            reflection_point_x= x + range * cos(phi)
            reflection_point_y= y + range* sin(phi)
            # Transfrom cell coordinates to cell indices.
            reflecting_cell= self.transform_point_to_grid_cell((reflection_point_x, reflection_point_y))
        return reflecting_cell

    
    @staticmethod
    def bresenham_line_drawing(start_grid_idx, end_grid_idx):
        '''Calculates all cell indices between start_grid_idx and end_grid_idx cell. 
        Input values are indices of first and last grid (line, column) (assuming integers).'''
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
        affected_cells.append(start_grid_idx)
        # Algorithm
        for i in range(fast_direction):
            err-= slow_direction
            if(err < 0):
                err+= fast_direction
                x+= ddx
                y+= ddy
            else:
                x+= pdx
                y+= pdy
            affected_cells.append((y, x))
        return affected_cells   
    

    def update_affected_cells(self, affected_cells):
        '''Get's a list of all effected cells by one beam. Decreases the logOdds values for
        all cells before the last cell. Increases the logOdds value for the last, reflecting, 
        cell.'''
        # Update the occupancy probability for each cell affected by the laser beam
        number_of_affected_cells= len(affected_cells)
        for i in range(number_of_affected_cells):
            cell_i, cell_j= affected_cells[i]
            old_log_odds_value= self.log_odds_map[cell_i][cell_j]
            # Bound the log_odds_values
            if(old_log_odds_value <= self.min_log_odds or old_log_odds_value >= self.max_log_odds):
                # self.log_odds_map[cell_i][cell_j]= old_log_odds_value    
                new_log_odds_value= old_log_odds_value
            else:
                # Decrease the occupancy for all cell that the ray passed
                if(i < (number_of_affected_cells - 1)):
                    new_log_odds_value= old_log_odds_value + self.log_odds_decreasing_probability
                # Increase the occupancy for the cell that reflected the cell
                else: 
                    new_log_odds_value= old_log_odds_value + self.log_odds_increasing_probability
            # Update grid cell
            self.log_odds_map[cell_i][cell_j]= new_log_odds_value


    def update_map(self, measurements, pose):
        '''Update the logOdds map by the given (x, y, heading) pose and (range, bearing) measurements. 
        Bounds the values of the logOdds map.'''
        x, y, heading= pose
        pose_i, pose_j= self.transform_point_to_grid_cell((x, y))
        for m in measurements:
            # Find grid cell that reflected the ray
            r, bearing = m 
            relfecting_cell= self.find_reflecting_grid_cell((r, bearing), pose)
            # Check if there was a reflecting cell 
            if(relfecting_cell):
                # Find all grid cells between pose and reflecting grid cell
                affected_cells= self.bresenham_line_drawing((pose_i, pose_j), (relfecting_cell))
                self.update_affected_cells(affected_cells)
    
    #_______________________________________________________________________________________________________________
    # Grid Cell manipulation
    #_______________________________________________________________________________________________________________

    def colorize_grid_black(self, grid_cell_indices):
        '''For testing. Change the color of the given grid cell to black.'''
        # Define log Odds value that correspond's to black
        logOdds_one= 100.0                               
        grid_idx_x, grid_idx_y= grid_cell_indices
        self.log_odds_map[grid_idx_x][grid_idx_y]= logOdds_one

    
    def colorize_grid_white(self, grid_cell_indices):
        '''For testing. Change the color of the given grid cell to white.'''
        # Define log Odds value that correspond's to black
        logOdds_zero= -100                               
        grid_idx_x, grid_idx_y= grid_cell_indices
        self.log_odds_map[grid_idx_x][grid_idx_y]= logOdds_zero


    def change_grid_cell_value(self, grid_cell_indices, value):
        '''Changes the value of the given grid to the given value.'''
        grid_idx_x, grid_idx_y= grid_cell_indices
        self.log_odds_map[grid_idx_x][grid_idx_y]= value


#__________________________________________________________________________________________________________________________
# OGMROSCommunication  
#__________________________________________________________________________________________________________________________

class OGMROSCommunication():
    def __init__(self, map_parameter, occupancy_parameter, ros_parameter, sensor_parameter):
        self.ogm= OccupancyGridMapping(map_parameter, occupancy_parameter, sensor_parameter)
        self.ogm.create_map()
        # Extract ros parameter
        link_state_topic, link_state_name, scan_topic, map_topic, self.update_rate= ros_parameter   
        # Define Subscriber and Publisher
        # Lock object to lock threads
        self.lock= threading.Lock()
        # Subscriber for odometry and scan data
        # Subscriber for link state -> pose
        self.link_state_message = None
        self.link_state_name = link_state_name
        self.link_state_index = None
        self.link_states_sunscriber = rospy.Subscriber(link_state_topic, LinkStates, self.link_state_callback)

        self.laser_scan_subscriber= rospy.Subscriber(scan_topic, LaserScan, self.laser_scan_callback)
        self.laser_scan= []
        self.map_publisher= rospy.Publisher(map_topic, LogOddsMap, queue_size=1) 


    # Callback functions and Message transformations
    #_______________________________________________________________________________________________________________
    
    def link_state_callback(self, link_states: LinkStates):
        '''Receive gazebo link state from topic.'''
        self.lock.acquire()
        # Extract message
        self.link_state_message = link_states

        # Find link state name index -> base_link index
        if self.link_state_index is None:
            try:
                self.link_state_index = link_states.name.index(self.link_state_name)
                rospy.loginfo(f"Found link state index: {self.link_state_index}")
            except ValueError:
                rospy.logwarn_throttle(5.0, f"Link {self.link_state_name} not found in Gazebo link states.")

        if self.link_state_index is None:
            for i in range(len(link_states.name)):
                if self.link_state_name == link_states.name[i]:
                    self.link_state_index = i
                    break
        
        self.lock.release()


    def laser_scan_callback(self, laser_scan):
        '''Receive laser scan from topic.'''
        self.lock.acquire()
        self.laser_scan= laser_scan
        self.lock.release()
    

    @staticmethod
    def transform_link_state_pose_to_planar_pose(link_state: LinkStates, link_state_index: int):
        '''
        Transforms the link state message to a planar pose, consisting of (x, y, yaw) tuple.
        '''
        link_state_pose: Pose = link_state.pose[link_state_index]

        x= link_state_pose.position.x
        y= link_state_pose.position.y
        orientation = link_state_pose.orientation
        # Transform quaternion angle's to euler angle's
        (roll, pitch, yaw)= euler_from_quaternion([orientation.x, orientation.y, orientation.z,
                                                orientation.w])
        planar_pose= (x, y, yaw)
        return planar_pose


    @staticmethod
    def transform_laser_scan_to_measurement(laser_scan: LaserScan):
        '''Transform the sensor msgs LaserScan to a list of measurement's consisting of 
        (range, bearing) tuple.'''
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


    def publish_occupancy_grid_message(self):
        '''Get's the logOdds map and do all necessary transformation's for publishing the Occupancy Message.'''
        # Publish current map from ogm algorithm
        self.map_publisher.publish(self.ogm.return_log_odds_map_object())


    def execute(self):
        '''Runs the occupancy grid mapping algorithm.'''
        update_rate= rospy.Rate(self.update_rate)
        while not rospy.is_shutdown():
            # rospy.loginfo("Mapping node running")
            # Check if data was received
            
            # Check if data is available
            if(self.link_state_message and self.link_state_index and self.laser_scan):
                rospy.loginfo_once("OGM Initalized.")

                # get data from callbacks
                self.lock.acquire()
                link_state = self.link_state_message
                link_state_index = self.link_state_index 
                laser_scan = self.laser_scan
                self.lock.release()
                
                # Transform data
                pose = self.transform_link_state_pose_to_planar_pose(
                    link_state=link_state,
                    link_state_index=link_state_index,
                )

                measurements = self.transform_laser_scan_to_measurement(laser_scan)

                # Increase map size if necessary
                extension_needed= True
                while(extension_needed):
                    extension_needed= self.ogm.map_extension_if_necessary(pose)

                # Update the map
                self.ogm.update_map(measurements, pose)
                # Transform and publish map
                self.publish_occupancy_grid_message()

            update_rate.sleep()


#__________________________________________________________________________________________________________________________
# Main  
#__________________________________________________________________________________________________________________________

def full_occupied_map(grid_map_obj):
    map_shape= np.shape(grid_map_obj.log_odds_map)
    grid_map_obj.log_odds_map= np.full((map_shape), 100.0)


def main():
     # Init Node
    rospy.init_node("optimized_occupancy_grid_algo_with_map_extension", anonymous=True)
    
    # Define map (size in mm)
    map_width_m= 10.0               # [m] -> 20 m
    map_height_m= 10.0              # [m] -> 20 m
    grid_resolution_m= 0.05         # [m] -> 50 mm grid resolution
    min_distance_to_border= 10.0    # The minimum distance from the actual robot pose to the border before extending the map
    
    # Define occupancy parameter
    prior_probability= 0.5          # Init map with probability of 0.5
    increasing_probability= 0.65
    decreasing_probability= 0.35
    max_log_odds= 100
    min_log_odds= -100
    
    # Define subscriber topics
    link_state_topic = "/gazebo/link_states"
    link_state_name = "robot_vacuum_cleaner::base_link"
    scan_topic= "scan"
    map_topic= "log_odds_map"
    
    # Define update rate of mapping algorithm
    update_rate= 12                 # Highest possible rate is 15
    
    # Define Sensor parameter
    min_sensor_range= 0.1
    max_sensor_range= 8.0
    
    # Summarize parameters
    map_parameter=[map_width_m, map_height_m, grid_resolution_m, min_distance_to_border]
    occupancy_parameter= [prior_probability, increasing_probability, decreasing_probability, min_log_odds, max_log_odds]
    ros_parameter= [link_state_topic, link_state_name, scan_topic, map_topic, update_rate]
    sensor_parameter= [min_sensor_range, max_sensor_range]

    # Initialize algorithm
    ros_ogm= OGMROSCommunication(map_parameter, occupancy_parameter, ros_parameter, sensor_parameter)

    # Start the algorithm
    ros_ogm.execute()

    


if __name__=="__main__":
    main()  