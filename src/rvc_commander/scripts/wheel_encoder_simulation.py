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
    '''
    Simulates the wheel encoder data, based on the pose of the robot. The pose is received from a topic, 
    which is expected to be published by a localization node. The simulated wheel encoder data is then
    published to a topic, which can be used by the controller node.
    '''
    def __init__(self, robot_width, link_state_name, link_state_topic= "gazebo/link_states", wheelencoder_topic="encoder", publish_rate= 10):
        # Robot parameter
        self.robot_width= robot_width
        # Init Wheel encoder message 
        self.distance= WheelEncoder()
        # Subscriber for link state -> pose
        self.link_state_message = None
        self.link_state_name = link_state_name
        self.link_state_index = None
        self.link_states_sunscriber = rospy.Subscriber(link_state_topic, LinkStates, self.link_state_callback)
        self.lock= threading.Lock()
        # Publisher
        self.publish_rate= publish_rate
        self.wheel_encoder_publisher= rospy.Publisher(wheelencoder_topic, WheelEncoder, queue_size=1)


    def link_state_callback(self, link_states):
        '''Receive pose from topic.'''
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


    @staticmethod
    def transform_link_state_pose_to_planar_pose(link_state: LinkStates, link_state_index: int):
        '''
        Transforms the geometry msgs pose to a planar pose, consisting of (x, y, yaw) tuple.
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
    def wheelencoder_simulation(old_pose, new_pose, width, eps_alpha= 1e-3):
        '''
        Get's the pose at x_t and x_t-1, as well as robot width and computes the distance the left 
        and right wheel traveled, since the last time stamp. 

        Parameters:
        ----------
        old_pose: tuple
            The pose at time x_t-1, given as (x, y, theta)
        new_pose: tuple
            The pose at time x_t, given as (x, y, theta)
        width: float
            The width of the robot, given as distance between the two wheels
        eps_alpha: float
            Threshold to determine if a turn took place, given as minimum angle in radians

        Returns:
        -------
        left_control: float
            The distance the left wheel traveled since the last time stamp
        right_control: float
            The distance the right wheel traveled since the last time stamp
        '''
        old_x, old_y, old_theta= old_pose
        new_x, new_y, new_theta= new_pose
        # Calculate alpha (turning angle)
        alpha= new_theta - old_theta
        alpha= atan2(sin(alpha), cos(alpha))

        # Compute direct distance between x_t and x_t-1
        d = sqrt((new_x-old_x)**2 + (new_y - old_y)**2)

        # If turning took place
        if(abs(alpha) > eps_alpha):
            # Calculate turning radius 
            radius = d / (2 * sin(alpha/2))
            # Calculate left and right control
            width_by_two= width / 2
            left_control= (radius - width_by_two) * alpha
            right_control= (radius + width_by_two) * alpha
        else:
            # If not turning took place
            left_control= d
            right_control= d
        return (left_control, right_control)        


    def execute(self):
        '''Main loop to simulate wheel encoder data and publish it.'''
        # Initialize Wheel Encoder Message 
        distance= WheelEncoder()
        old_pose= None
        rate= rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            # Check if data was received
            if(self.link_state_message and self.link_state_index is not None):
                # Copy pose
                self.lock.acquire()
                link_state = self.link_state_message
                self.lock.release()
                # Pose -> planar pose
                new_pose= self.transform_link_state_pose_to_planar_pose(
                    link_state=link_state,
                    link_state_index=self.link_state_index
                )
                # Check if old pose was initialized already
                if(old_pose):
                    rospy.loginfo_once("Wheel encoder simulation initialized.")
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
    link_state_topic= "gazebo/link_states"
    link_state_name = "robot_vacuum_cleaner::base_link"
    wheelencoder_topic= "wheel_encoder" 
    publish_rate= 10
    wheel_encoder_simulation= SimulateWheelEncoder(robot_width= wheel_separation, link_state_name= link_state_name, link_state_topic=link_state_topic, 
                                wheelencoder_topic= wheelencoder_topic, publish_rate= publish_rate)
    # Execute 
    wheel_encoder_simulation.execute()



if __name__=="__main__":
    main()