#!/usr/bin/env python3

import rospy 
from sensor_msgs.msg import LaserScan

import numpy as np  


class My_Laser_Scan_Data_Transformation():
    def __init__(self, laser_scan_topic):
        self.measurements= []
        self.laser_scan_subscriber= rospy.Subscriber(laser_scan_topic, LaserScan, self.transform_laser_scan_data)


    def transform_laser_scan_data(self, laser_scan_data):
        '''Reads the LaserScan message from the given scan topic and transforms it to 
        a list of range and angle tuples.'''
        # Initialization
        min_angle= laser_scan_data.angle_min
        angle_increment= laser_scan_data.angle_increment
        angle= min_angle
        self.measurements= []
        # Transform LaserScan data
        for range in laser_scan_data.ranges:
            measurement= (range, angle)
            angle+= angle_increment
            self.measurements.append(measurement)
            # Write data to console
            rospy.loginfo("Range and bearing: %f %f", range, (angle - angle_increment))
        

    @staticmethod
    def execute():
        rospy.spin()




def main():
    rospy.init_node("laser_scan_transformer", anonymous=True)
    laser_scan_topic= "scan"
    scan_transformator= My_Laser_Scan_Data_Transformation(laser_scan_topic)
    scan_transformator.execute()


if __name__=="__main__":
    main()