#!/usr/bin/env python3

from dataclasses import dataclass

import rospy
import threading
import numpy as np
from math import exp
from nav_msgs.msg import OccupancyGrid
from rbpf_slam.msg import LogOddsMap


@dataclass
class ROSParams:
    log_odds_topic= "log_odds_map"
    map_frame= "map"
    map_topic= "map"
    publish_rate= 1



class MapTransformationNode:
    def __init__(
        self,
        ros_params: ROSParams,
    ):
        # Initialize ROS node
        self.ros_params = ros_params
        self.lock = threading.Lock()

        self.map_msg:LogOddsMap = None

        # Def subscriber
        self.map_sub = rospy.Subscriber(
            name=self.ros_params.log_odds_topic,
            data_class=LogOddsMap,
            callback=self.map_cb,
        )

    def map_cb(self, log_odds_map: LogOddsMap):
        with self.lock:
            self.map_msg = log_odds_map


    def exe(self):
        publish_rate = rospy.Rate(self.ros_params.publish_rate)
        while not rospy.is_shutdown():
            with self.lock:
                pass


            publish_rate.sleep()
        


def main():
    rospy.init_node("map_transformation_node", anonymous= True)
    
    ros_params = ROSParams()
    node = MapTransformationNode(ros_params=ros_params)


if __name__=="__main__":
    main()