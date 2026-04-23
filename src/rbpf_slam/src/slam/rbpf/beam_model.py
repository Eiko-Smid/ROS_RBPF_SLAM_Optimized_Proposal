from typing import List, Tuple
import numpy as np
from scipy.stats import norm as normal_dist

from .measurement_model import MeasurementModel
from slam.scan_matcher.scan_matcher import ScanMatcher
from slam.infrastructure.defs import Pose2D


class BeamModel(MeasurementModel):
    def __init__(self, measurement_model_parameter, sensor_parameter):
        self.sigma_hit, self.z_hit, self.z_short, self.z_max, self.z_random, self.lambda_short= measurement_model_parameter
        self.min_sensor_range, self.max_sensor_range= sensor_parameter


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
        phi= np.arctan2(np.sin(heading + bearing), np.cos(heading + bearing))
        reflection_point_x= x + range * np.cos(phi)
        reflection_point_y= y + range* np.sin(phi)
        
        # Transfrom cell coordinates to cell indices.
        reflecting_cell= self.transform_point_to_grid_cell((reflection_point_x, reflection_point_y))
        
        return reflecting_cell


    def find_occupied_grid_cell(self, start_grid_idx, end_grid_idx, scan_matcher: ScanMatcher):
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
            
            if(scan_matcher.ogm.is_cell_inside_map((y, x))):
                is_inside= True
                # Check if cell is occupied
                occupancy_value= scan_matcher.ogm.occupancy_grid_map[y][x]
                if(occupancy_value == self.occ):
                    leaf_loop= True
                    is_occupied= True
            else: 
                is_inside= False
                leaf_loop= True
        
        return affected_cells, (y, x), is_occupied, is_inside
    
    
    def predict_measurement(self, particle, particle_cell_index, measurement, scan_matcher: ScanMatcher, max_sensor_range):
        '''Predict the measurement based on the given (x, y, theta) particle pose and the (range, bearing) measurement. 
        The given particle must be inside the map. The measurement needs to be a valid measurement (not inf for example and 
        in range parameter of sensor). Returns the predicted range between the particle pose and the grid cell, which should 
        have reflected the beam, if the particle is the true pose of the robot. 
        If the algorithm detects a cell which occupancy value is "unknown" or if it detects a cell which is outside the map,
        these cell will be treated as if it's occupancy value is "occupied".'''
        # Extract data
        px, py, theta= particle
        range, bearing= measurement
        
        # Compute grid cell indices of max measurement
        max_measurement_grid_cell= self.find_reflecting_grid_cell((max_sensor_range, bearing), particle)
        
        # Estimate reflecting grid cell 
        affected_cells, last_cell, is_occupied, is_inside= self.find_occupied_grid_cell(particle_cell_index, max_measurement_grid_cell)
        
        # Is cell occupied or unknown or out of map or max range, treat the cell as if the cell is occupied and predict the measurement
        rx, ry= scan_matcher.ogm.transform_grid_cell_to_point(last_cell)
        
        # Calculate range and bearing of reflecting cell
        dx= rx - px
        dy= ry - py
        predicted_range= np.sqrt(dx**2 + dy**2)
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
            probability+= self.z_short * self.lambda_short * np.exp((-self.lambda_short) * measured_range)
        
        # Take measurement failures into account
        if(measured_range == self.max_sensor_range):
            probability+= self.z_max * 1.0
        
        # Take random measurements into account
        if(measured_range < self.max_sensor_range):
            probability+= self.z_random * 1/self.max_sensor_range
        
        return probability


    def likelihood(
            self,
            pose: Pose2D,
            measurements: List[Tuple[float, float]],
            scan_matcher: ScanMatcher,
    ):
        # Transform pose to gird cell
        x, y, thata = pose
        pose_cell = scan_matcher.ogm.transform_point_to_grid_cell((x, y))

        # Check if cell is inside map
        if scan_matcher.ogm.cell_inside_map(pose_cell):
            for m in measurements:
                # Predict the measurement
                predicted_measurement, is_inside, is_occupied= self.predict_measurement(pose, pose_cell, m)                        
                w= self.calculate_measurement_probability(m, predicted_measurement)
                measurement_weight+= w * w * w
            # Update particle measurement_weight
            particle_weight*= measurement_weight
        else:
            # Make sure particle outside map will be erased
            particle_weight= 0.0  

        return particle_weight