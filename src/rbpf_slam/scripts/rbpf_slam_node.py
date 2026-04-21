#!/usr/bin/env python3

import rospy

from slam.rbpf_slam import RbpfSlam


class RbpfSlamNode:
    def __init__(self):
        self._slam = RbpfSlam()
        rospy.loginfo("rbpf_slam_node initialized")

    def spin(self):
        self._slam.start()
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("rbpf_slam_node")
    node = RbpfSlamNode()
    node.spin()
