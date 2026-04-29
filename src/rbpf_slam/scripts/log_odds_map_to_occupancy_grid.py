#!/usr/bin/env python3

import rospy
import threading
import numpy as np
from math import exp
from nav_msgs.msg import OccupancyGrid
from rvc_commander.msg import LogOddsMap


'''
Description:

    This program receives a logOdds map and transforms the map to an OccupancyGrid map message.
    After the transformation, is map is being published to the map topic.
'''

class LogOddsToOccupancyGrid():
    def __init__(self, log_odds_topic, map_frame, map_topic, publish_rate):
        # LogOdds map message object
        self.log_odds_map_object= None
        self.log_odds_map= None
        self.occupancy_grid_msg= OccupancyGrid()
        self.occupancy_grid_msg.header.frame_id= map_frame
        self.number_of_grid_cells= None
        # Occupancy Grid Map 
        # Lock object 
        self.lock= threading.Lock()
        # Subscriber 
        self.log_odds_map_subscriber= rospy.Subscriber(log_odds_topic, LogOddsMap, self.log_odds_map_callback_v2)
        # Publisher
        self.occupancy_grid_publisher_object= rospy.Publisher(map_topic, OccupancyGrid, queue_size=1)
        self.publish_rate= publish_rate
        # Flags
        self.log_odds_msg_initialized= False
        # rospy.loginfo("\n\nInit Done\n\n")


    def log_odds_map_callback(self, log_odds_map):
        '''Receives the log odds map'''
        self.lock.acquire()
        self.log_odds_map_object= log_odds_map
        if(log_odds_map.data and not self.log_odds_msg_initialized):
            # Init occupancy grid message 
            self.occupancy_grid_msg.info.width= log_odds_map.info.width
            self.occupancy_grid_msg.info.height= log_odds_map.info.height
            self.occupancy_grid_msg.info.origin.position.x= log_odds_map.info.origin.position.x
            self.occupancy_grid_msg.info.origin.position.y= log_odds_map.info.origin.position.y
            self.occupancy_grid_msg.info.origin.orientation = log_odds_map.info.origin.orientation
            self.occupancy_grid_msg.info.resolution= log_odds_map.info.resolution
            # Calculate number of grid cells
            self.number_of_grid_cells= int(log_odds_map.info.width * log_odds_map.info.height)
            # Set initialization flag to true
            self.log_odds_msg_initialized= True
            # rospy.loginfo("\n\nInit Done\n\n")
        self.lock.release()
    

    def log_odds_map_callback_v2(self, log_odds_map):
        '''Receives the log odds map'''
        self.lock.acquire()
        self.log_odds_map_object= log_odds_map
        if(log_odds_map.data):
            # Init occupancy grid message 
            self.occupancy_grid_msg.info.width= log_odds_map.info.width
            self.occupancy_grid_msg.info.height= log_odds_map.info.height
            self.occupancy_grid_msg.info.origin.position.x= log_odds_map.info.origin.position.x
            self.occupancy_grid_msg.info.origin.position.y= log_odds_map.info.origin.position.y
            self.occupancy_grid_msg.info.origin.orientation = log_odds_map.info.origin.orientation
            self.occupancy_grid_msg.info.resolution= log_odds_map.info.resolution
            self.occupancy_grid_msg.header.frame_id = log_odds_map.header.frame_id
            # Calculate number of grid cells
            self.number_of_grid_cells= int(log_odds_map.info.width * log_odds_map.info.height)            
            # rospy.loginfo("\n\nInit Done\n\n")
        self.lock.release()    


    @staticmethod
    def log_odds_to_probability(log_odds):
        '''Calculates the probability according to the given log Odds value.'''
        log_odds_exp= exp(log_odds)
        return log_odds_exp / (1+ log_odds_exp)


    def occupancy_grid_publisher(self):
        '''First transform the logOdds map to probability space. Then it multiplies each
        values by 100 to get the correct form of the OccupancyGrid.data. At least it publishes 
        the scaled probability map as an OccupancyGrid Message.'''  
        # Transform logOdds values to probability's
        probability_grid_map= np.copy(self.log_odds_map.data)
        for i in range(self.number_of_grid_cells):
            log_odds_value= probability_grid_map[i] 
            probability_grid_map[i]= self.log_odds_to_probability(log_odds_value)

        # Scale values and copy the map to the message object (data type must be int8)
        self.occupancy_grid_msg.data= np.copy(probability_grid_map * 100).astype(dtype= np.int8)
        
        # Publish the message 
        self.occupancy_grid_msg.header.stamp= rospy.Time.now()
        self.occupancy_grid_publisher_object.publish(self.occupancy_grid_msg)

    
    def execute(self):
        publish_rate= rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            # Check if logOdds map was received
            if(self.log_odds_map_object):
                rospy.loginfo("\nLogOdds map received.")

                # Extract map
                self.lock.acquire()
                self.log_odds_map = self.log_odds_map_object
                self.lock.release()

                # Publish map
                self.occupancy_grid_publisher()
            else:
                rospy.loginfo("\nNo logOdds map received")
            publish_rate.sleep()
        

def main():
    rospy.init_node("log_odds_map_to_occupancy_grid", anonymous= True)
    log_odds_topic= "log_odds_map"
    map_frame= "map"
    map_topic= "map"
    publish_rate= 1
    transform_occupancy_grid_map= LogOddsToOccupancyGrid(log_odds_topic= log_odds_topic, map_frame= map_frame, map_topic= map_topic,
                                                            publish_rate= publish_rate)
    transform_occupancy_grid_map.execute()


if __name__=="__main__":
    main()