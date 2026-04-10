#!/usr/bin/env python3

import rospy
from typing import List, Tuple, Optional
import threading
import numpy as np
from math import exp, atan2, sin, cos, radians, degrees, floor, ceil, isfinite, log
import time
from geometry_msgs.msg import Pose, Point
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
# from nav_msgs.msg import OccupancyGrid
from rvc_commander.msg import Measurement
from rvc_commander.msg import LogOddsMap



class OGM:
    '''
    Implementation of the occupancy grid mapping algorithm. The map is represented in log Odds space. The map is
    initialized with a prior probability.
    
    Attention! After creating
    the grid map with "create_map()" the map will stay in log Odds space forever. To get the occupancy grid map
    out of the self.log_odds_map, the two methodes "transform_log_odds_map_to_occupancy_grid_map" and 
    "transform_log_odds_map_to_probability_map" need to be used.
    '''
    IDX_X= 0
    IDX_Y= 1
    def __init__(self, map_parameter: List[float], occupancy_parameter: List[float], sensor_parameter: List[float]) -> None:
        '''
        Constructor of the OGM class. Initializes all parameters and variables needed for the algorithm. Also checks 
        if the given parameters are valid and sets them to default values if they are not valid.

        Parameters:
        ----------
        map_parameter: List of float
            All map parameters needed. Containing of the minimum distance to the border of the map, where the map should 
            be extended.
        occupancy_parameter: List of float
            All occupancy parameters needed. Containing of the prior probability, the increasing probability, the decreasing
            probability, the minimum log Odds value, and the maximum log Odds value.
        sensor_parameter: List of float
            All sensor parameters needed. Containing of the minimum and maximum sensor range.

        '''
        # Extract parameter
        self.min_distance_to_border= map_parameter
        prior_probability, increasing_probability, decreasing_probability, self.min_log_odds, self.max_log_odds= occupancy_parameter
        self.min_sensor_range, self.max_sensor_range= sensor_parameter
        self.grid_resolution_m = None

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
            self.log_odds_prior= np.log(prior_probability / (1 - 0.5))    
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


    def init_map(self, map_width: float, map_height: float, grid_resolution: float) -> None:
        '''Create map and init prior probability'''
        if map_width <= 0 or map_height <= 0 or grid_resolution <= 0:
            rospy.loginfo("\nThe map width, map height, and grid resolution must be positive values.\n")
            rospy.loginfo("The map was not initialized.")
            return

        # Init map parameters
        self.map_width_m = map_width
        self.map_height_m = map_height
        self.grid_resolution_m = grid_resolution

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
        self.update_log_odds_message()
        
        # Define the border values for the map
        half_map_width= self.map_width_m / 2.0
        half_map_height= self.map_height_m / 2.0
        self.left_map_border_m= -half_map_width 
        self.top_map_border_m= half_map_height
        self.right_map_border_m= half_map_width
        self.bottom_map_border_m= - half_map_height        

        rospy.loginfo("An empty map was successfully initialized from the given parameters.")
        rospy.loginfo(
            f"Map width= {self.map_width_m}, Map height= {self.map_height_m},"
            f" Number of cells in x direction= {self.number_of_cells_x},"
            f" Number of cells in y direction= {self.number_of_cells_y}"
        )


    def init_map_from_map(self, log_odds_map: np.ndarray, grid_resolution: float) -> None:
        '''Create the map from a given map.'''
        self.grid_resolution_m = grid_resolution

        if log_odds_map is None:
            rospy.loginfo("The given map is None. The map was not initialized.")
            return

        self.log_odds_map = np.array(log_odds_map, copy=True)

        pad_y = 1 if self.log_odds_map.shape[0] % 2 == 0 else 0
        pad_x = 1 if self.log_odds_map.shape[1] % 2 == 0 else 0

        if pad_y or pad_x:
            self.log_odds_map = np.pad(
                self.log_odds_map,
                pad_width=((pad_y, 0), (pad_x, 0)),
                mode="constant",
                constant_values=self.log_odds_prior,
            )

        self.number_of_cells_y, self.number_of_cells_x = self.log_odds_map.shape

        self.map_width_m = self.number_of_cells_x * self.grid_resolution_m
        self.map_height_m = self.number_of_cells_y * self.grid_resolution_m

        self.shift_x = self.map_width_m / 2
        self.shift_y = self.map_height_m / 2

        self.update_log_odds_message()

        half_map_width = self.map_width_m / 2.0
        half_map_height = self.map_height_m / 2.0
        self.left_map_border_m = -half_map_width
        self.top_map_border_m = half_map_height
        self.right_map_border_m = half_map_width
        self.bottom_map_border_m = -half_map_height

        rospy.loginfo("The map was successfully initialized from the given map.")
        rospy.loginfo(
            f"Map width= {self.map_width_m}, Map height= {self.map_height_m},"
            f" Number of cells in x direction= {self.number_of_cells_x},"
            f" Number of cells in y direction= {self.number_of_cells_y}"
        )


    def return_log_odds_map(self) -> np.ndarray:
        '''Returns the grid map in log odds form.'''
        return self.log_odds_map

    
    def return_log_odds_map_object(self) -> LogOddsMap:
        '''
        Returns a log odds map message object containing the map and the map metadata.
        '''
        # Copy the logOdds map to the message
        self.log_odds_map_msg.data= self.log_odds_map.ravel()
        # generate timestamp
        self.log_odds_map_msg.header.stamp= rospy.Time.now()        
        return self.log_odds_map_msg

    
    def extend_map(self, direction: str, distance: float) -> Tuple[int, bool]:
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


    def map_extension_if_necessary(self, pose: Tuple[float, float, float]) -> bool:
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


    def update_log_odds_message(self) -> None:
        self.log_odds_map_msg.info.width= self.number_of_cells_x
        self.log_odds_map_msg.info.height= self.number_of_cells_y
        origin_x, origin_y= self.transform_grid_cell_to_point((0, 0))
        self.log_odds_map_msg.info.origin.position.x= origin_x
        self.log_odds_map_msg.info.origin.position.y= origin_y
        self.log_odds_map_msg.info.resolution= self.grid_resolution_m


    #_______________________________________________________________________________________________________________
    # Transformations
    #_______________________________________________________________________________________________________________

    @staticmethod
    def log_odds_to_probability(log_odds: float) -> float:
        '''Calculates the probability according to the given log Odds value.'''
        log_odds_exp= exp(log_odds)
        return log_odds_exp / (1+ log_odds_exp)
    

    @staticmethod
    def transform_occupany_map_to_log_odds_map(ogm: np.ndarray, occ_params: tuple, log_odds_param: tuple) -> float:
        '''
        Transforms the given occupancy grid map to a log Odds map.  
        '''
        # Extract parameters
        occ, free = occ_params
        log_odds_occ, log_odds_free, log_odds_unknown = log_odds_param 

        # Assign log odds values to log odds map
        log_odds_map = np.full_like(ogm, log_odds_unknown, dtype=float)
        log_odds_map[ogm == occ] = log_odds_occ
        log_odds_map[ogm == free] = log_odds_free

        return log_odds_map
    

    @staticmethod
    def probability_to_occupancy(probability: float) -> float:
        occupancy_value= 0.0
        if(probability < 0.5):
            occupancy_value= 0.0
        elif(probability == 0.5):
            occupancy_value= 0.5
        else:
            occupancy_value= 1.0
        return occupancy_value


    @staticmethod
    def transform_log_odds_map_to_probability_map(log_odds_map: np.ndarray) -> np.ndarray:
        '''Transfers the map from the log Odds space to the 
        probability space and returns the transformened map.'''
        probability_map= np.copy(log_odds_map)
        map_shape= np.shape(probability_map)
        for i in range(map_shape[0]):
             for j in range(map_shape[1]):
                #  probability_map[i][j]= self.log_odds_to_probability(self.log_odds_map[i][j])
                probability_map[i][j]= OGM.log_odds_to_probability(log_odds_map[i][j])
        return np.copy(probability_map)

    
    @staticmethod
    def transform_probability_map_to_occupancy_map(probability_grid_map: np.ndarray) -> np.ndarray:
        '''Transforms the given grid map from probability space to occupancy space.'''
        occupancy_grid_map= np.copy(probability_grid_map)
        map_shape= np.shape(probability_grid_map)
        # Round probability of each grid cell to 0, 1, or 0.5
        for i in range(map_shape[0]):
            for j in range(map_shape[1]):
                occupancy_grid_map[i][j]= OGM.probability_to_occupancy(probability_grid_map[i][j])
        return occupancy_grid_map

    
    @staticmethod
    def transform_log_odds_map_to_occupancy_grid_map(log_odds_map: np.ndarray) -> np.ndarray:
        '''Transforms the given map from log odds space to occupancy space.'''
        probability_grid_map= OGM.transform_log_odds_map_to_probability_map(log_odds_map)
        occupancy_grid_map= OGM.transform_probability_map_to_occupancy_map(probability_grid_map)
        return occupancy_grid_map


    def transform_point_to_grid_cell(self, point: Tuple[float, float]) -> Tuple[int, int]:
        '''Transforms an (x, y) point to the array access indices (i, j for row, column). '''
        x,y = point
        x_shifted= x + self.shift_x
        y_shifted= y + self.shift_y
        i= floor(y_shifted/self.grid_resolution_m)
        j= floor(x_shifted/self.grid_resolution_m)
        return (i, j)


    def transform_grid_cell_to_point(self, grid_cell: Tuple[int, int]) -> Tuple[float, float]:
        '''Transforms the given grid cell (i, j) to a (x, y) point in the real world.'''
        i, j= grid_cell
        x= j * self.grid_resolution_m - self.shift_x + self.grid_resolution_m/2
        y= i * self.grid_resolution_m - self.shift_y + self.grid_resolution_m/2
        return (x, y)    



    #_______________________________________________________________________________________________________________
    # Main Algorithm
    #_______________________________________________________________________________________________________________

    def find_reflecting_grid_cell(self, measurement: Tuple[float, float], pose: Tuple[float, float, float]) -> Optional[Tuple[int, int]]:
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
    def bresenham_line_drawing(start_grid_idx: Tuple[int, int], end_grid_idx: Tuple[int, int]) -> List[Tuple[int, int]]:
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
    

    def update_affected_cells(self, affected_cells: List[Tuple[int, int]]) -> None:
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


    def update_map(self, measurements: List[Tuple[float, float]], pose: Tuple[float, float, float]) -> None:
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
    # Map extraction
    #_______________________________________________________________________________________________________________

    def get_neighbors(self, cell) -> List[Tuple[int, int]]:
        '''
        Get's the eight neighbors valures of the given cell. 
        '''
        i, j = cell
        sub_map = self.log_odds_map[i-1:i+2, j-1:j+2].copy()
        sub_map = sub_map.ravel()

        neighbors = np.delete(sub_map, 4)

        return neighbors
        

    def cell_inside_map(self, cell):
        '''
        Check if cell is inside the logOdds map. 
        Useful when computing indices and not sure if index is inside arr or not.
        '''
        cell_inside = False
        i, j = cell

        if 0 <= i < self.log_odds_map.shape[0] and 0 <= j < self.log_odds_map.shape[1]:
            cell_inside = True

        return cell_inside
 

    def cell_belongs_to_surface(self, cell, free_thresh=-2.0, min_free_count=2):
        '''
        Check if the given cell belongs to a surface. A cell belongs to a surface if it has at least "min_free_count"
        free neighbors. If all values around are occupied the cell doesn't belong to a surface, even if the cell itself
        is occupied. This is because the cell would be in the middle of an object and not on the surface of an object. 
        '''
        free_count = 0
        belongs_to_surface = False
        
        neighbors = self.get_neighbors(cell)

        for logOdds in neighbors:
            if logOdds < free_thresh:
                free_count += 1

        if free_count >= min_free_count:
            belongs_to_surface = True
        
        return belongs_to_surface


    def extract_map_for_scan_matching(self, pose, radius, delta_r=1.0, occ_thresh=2.0) -> np.ndarray:
        '''
        This method extracts a part of the map which is used as a target pointcloud for scan matching. 
        '''
        # TODO: Ensure that extracted map size is not too big. Right now it can happen that we are at teh border of the 
        # map array and accidentally jump over and extract a huge part of the map. This is especially the case fpr the method
        # "cell_belongs_to_surface".
        valid_points = []

        # Convert radius into cell numbers
        r_cells = ceil((radius + delta_r) / self.grid_resolution_m)
        r_cells_squared = r_cells * r_cells

        # Transform pose into grid cell
        i_pose, j_pose = self.transform_point_to_grid_cell(pose[:2])

        # With the for loops we define a general square with center cell and it has the size of the radius*2 + 1 
        for di in range(-r_cells, r_cells + 1):
            for dj in range(-r_cells, r_cells + 1):

                # For every cell in the general square we check if the cell is inside the radius (from the center point)
                # Using squared values to avoid sqrt -> faster
                # Skip if not inside circle area
                if di * di + dj * dj > r_cells_squared:
                    continue
                
                # Compute the actual cell indices in the map/array from our general square and the center point
                i = i_pose + di
                j = j_pose + dj

                # Check if cell is indeed inside our map 
                if not self.cell_inside_map((i, j)):
                    continue
                
                # Check if cell is occupied -> extract point coordinates
                if self.log_odds_map[i, j] < occ_thresh:
                    continue

                # Check if cell belongs to surface
                if not self.cell_belongs_to_surface(
                    cell=(i,j),
                    free_thresh=-2.0,
                    min_free_count=2,
                ):
                    continue
                
                # Transform cell to point and append 
                x, y = self.transform_grid_cell_to_point((i, j))
                valid_points.append((x, y))
                
        return np.copy(valid_points)
        

    #_______________________________________________________________________________________________________________
    # Grid Cell manipulation
    #_______________________________________________________________________________________________________________

    def colorize_grid_black(self, grid_cell_indices: Tuple[int, int]) -> None:
        '''For testing. Change the color of the given grid cell to black.'''
        # Define log Odds value that correspond's to black
        logOdds_one= 100.0                               
        grid_idx_x, grid_idx_y= grid_cell_indices
        self.log_odds_map[grid_idx_x][grid_idx_y]= logOdds_one

    
    def colorize_grid_white(self, grid_cell_indices: Tuple[int, int]) -> None:
        '''For testing. Change the color of the given grid cell to white.'''
        # Define log Odds value that correspond's to black
        logOdds_zero= -100                               
        grid_idx_x, grid_idx_y= grid_cell_indices
        self.log_odds_map[grid_idx_x][grid_idx_y]= logOdds_zero


    def change_grid_cell_value(self, grid_cell_indices: Tuple[int, int], value: float) -> None:
        '''Changes the value of the given grid to the given value.'''
        grid_idx_x, grid_idx_y= grid_cell_indices
        self.log_odds_map[grid_idx_x][grid_idx_y]= value