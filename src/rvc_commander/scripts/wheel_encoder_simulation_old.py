#!/usr/bin/env python3

import rospy

# Python library's 
from math import sqrt, sin, cos, atan2
import threading
# Messages
from geometry_msgs.msg import Pose, Twist
from gazebo_msgs.msg import LinkStates
from tf.transformations import euler_from_quaternion
from rvc_commander.msg import WheelEncoder



class SimulateWheelEncoder():
    def __init__(self, robot_width, odom_topic= "odom", wheelencoder_topic="encoder", publish_rate= 10):
        # Robot parameter
        self.robot_width= robot_width
        # Init Wheel encoder message 
        self.distance= WheelEncoder()
        # Subscriber for odometry 
        self.odom_subscriber= rospy.Subscriber(odom_topic, Pose, self.pose_callback)
        self.lock= threading.Lock()
        self.geometry_pose= None
        # Publisher
        self.publish_rate= publish_rate
        self.wheel_encoder_publisher= rospy.Publisher(wheelencoder_topic, WheelEncoder, queue_size=1)


    def pose_callback(self, pose):
        '''Receive pose from topic.'''
        self.lock.acquire()
        self.geometry_pose= pose
        self.lock.release()


    @staticmethod
    def transform_pose_to_planar_pose(pose):
        '''Transforms the geometry msgs pose to a planar pose, consisting of 
        (x, y, yaw) tuple.'''
        x= pose.position.x
        y= pose.position.y
        # Transform quaternion angle's to euler angle's
        (roll, pitch, yaw)= euler_from_quaternion([pose.orientation.x, pose.orientation.y, pose.orientation.z,
                                                pose.orientation.w])
        planar_pose= (x, y, yaw)
        return planar_pose


    @staticmethod
    def wheelencoder_simulation(old_pose, new_pose, width):
        old_x, old_y, old_theta= old_pose
        new_x, new_y, new_theta= new_pose
        # Calculate alpha (turning angle)
        alpha= new_theta - old_theta
        alpha= atan2(sin(alpha), cos(alpha))
        # If turning took place
        if(alpha != 0):
            # Calculate turning radius 
            radius= (new_x - old_x) / (sin(old_theta + alpha) - sin(old_theta))
            # Calculate left and right control
            width_by_two= width/2
            left_control= (radius - width_by_two) * alpha
            right_control= (radius + width_by_two) * alpha
        else:
            # If not turning took place
            distance= sqrt((new_x - old_x)**2 + (new_y - old_y)**2)
            left_control= distance
            right_control= distance
        return (left_control, right_control)        


    def execute(self):
        # Initialize Wheel Encoder Message 
        distance= WheelEncoder()
        old_pose= None
        rate= rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            # Check if data was received
            if(self.geometry_pose):
                # Copy pose
                self.lock.acquire()
                pose= self.geometry_pose
                self.lock.release()
                # Pose -> planar pose
                new_pose= self.transform_pose_to_planar_pose(pose)
                # Check if old pose was initialized already
                if(old_pose):
                    # Simulate encoder data
                    left_control, right_control= self.wheelencoder_simulation(old_pose, new_pose, self.robot_width)
                    # Publish wheel distances 
                    distance.left= left_control
                    distance.right= right_control
                    self.wheel_encoder_publisher.publish(distance)
                # Save values 
                old_pose= new_pose
            rate.sleep()


def main():
    # Start node
    rospy.init_node("wheel_encoder_simulation_node", anonymous= True)
    # Calculate wheel separation
    h_chassis= 0.15
    dist_chassis_to_ground= h_chassis/5
    r_wheel= h_chassis/2 + dist_chassis_to_ground
    w_wheel= 0.3 * r_wheel
    r_chassis= 0.25
    wheel_separation= 2 * r_chassis + w_wheel
    # Init SimulateWheelEncoder
    odom_topic= "true_odom"
    wheelencoder_topic= "wheel_encoder" 
    publish_rate= 10
    wheel_encoder_simulation= SimulateWheelEncoder(robot_width= wheel_separation, odom_topic= odom_topic, 
                                wheelencoder_topic= wheelencoder_topic, publish_rate= publish_rate)
    # Execute 
    wheel_encoder_simulation.execute()



if __name__=="__main__":
    main()