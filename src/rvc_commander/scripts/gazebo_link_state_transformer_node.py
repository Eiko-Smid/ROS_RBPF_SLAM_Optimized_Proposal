#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose, Twist
from gazebo_msgs.msg import LinkStates
import threading


class Link_State_Transformer():
    '''Get's the data from the given Topic "link_states_topic" and transforms the data to geometry_msgs type.
    The transformed messages are the pose (geometry_msgs.msg Pose) and the velocity (geometry_msgs.msg Twist)
    of the given link name. After the data was transformed, it will be published to the given topics 
    "odom_topic" (pose) and "velocity_topic" (velocity).'''
    def __init__(self, odom_topic, velocity_topic, link_states_topic, link_name, publish_rate):
        self.link_name= link_name
        self.link_state_index= None
        self.link_state_message= None
        # Define subscriber to subscribe from pose
        self.subscriber= rospy.Subscriber(link_states_topic, LinkStates, self.read_link_data)
        # Publisher for publishing extracted pose as geometry_msgs.msg Pose
        self.odom_publisher= rospy.Publisher(odom_topic, Pose, queue_size=1)
        # self.velocity_publisher= rospy.Publisher(velocity_topic, Twist, queue_size=10)
        self.publish_rate= publish_rate
        # Lock object to block threads
        self.lock= threading.Lock()


    def read_link_data(self, link_states):
        '''Reads the link state message data from the topic self.subscriber_topic and
        saves the message in member variable. Also finds index of link.'''
        # Lock so that main thread is unable to suspend this thread before finish 
        # receiving the whole message. 
        self.lock.acquire()
        # Save message
        self.link_state_message= link_states             
        # Find index of desired link
        for i in range(len(link_states.name)):
            if(self.link_name == link_states.name[i]):
                self.link_state_index= i
        self.lock.release()
        # self.boolean_publisher.publish(1)


    @staticmethod
    def transform_link_state_pose_to_pose_direct(link_state_pose):
        '''Transforms the gazebo link state Pose to geometry message pose. Works
        because gazebo massage includes geometry pose.'''
        # Extract pose and orientation
        pose= Pose()
        pose.position= link_state_pose.position
        pose.orientation= link_state_pose.orientation
        return pose


    @staticmethod
    def transform_link_state_pose_to_pose_indicrect(link_state_pose):
        '''Transforms the gazebo link state Pose to geometry message pose.
        Not needed anymore. Just to show how to extract message in general.'''
        # Extract pose and orientation
        x= link_state_pose.position.x
        y= link_state_pose.position.y
        z= link_state_pose.position.z
        angular_x= link_state_pose.orientation.x
        angular_y= link_state_pose.orientation.y
        angular_z= link_state_pose.orientation.z
        angular_w= link_state_pose.orientation.w
        pose= Pose()
        pose.position.x= x
        pose.position.y= y
        pose.position.z= z
        pose.orientation.x= angular_x
        pose.orientation.y= angular_y
        pose.orientation.z= angular_z
        pose.orientation.w= angular_w
        return pose

    
    @staticmethod
    def transform_link_state_twist_to_twist(link_state_twist):
        '''Transform the gazebo Twist message to geometry_msgs Twist.Works
        because gazebo massage includes geometry twist.'''
        twist= Twist()
        twist= link_state_twist
        return twist

   
    def execute(self):
        '''Run transformation.'''
        rate= rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            # Check if message was received for the first time
            if(self.link_state_index):
                # Copy Message while blocking callback thread
                self.lock.acquire()
                link_state_message= self.link_state_message
                self.lock.release()          
                # Extract pose and twist 
                link_state_pose= link_state_message.pose[self.link_state_index]
                link_state_twist= link_state_message.twist[self.link_state_index]
                # Transform pose and velocity
                pose= self.transform_link_state_pose_to_pose_direct(link_state_pose)
                velocity= self.transform_link_state_twist_to_twist(link_state_twist)
                # Publish Pose and twist
                self.odom_publisher.publish(pose)
                # self.velocity_publisher.publish(velocity)
            rate.sleep()


def main():
    # Init Node
    rospy.init_node("gazebo_link_state_transformer_node", anonymous=True)
    link_states_topic= "gazebo/link_states"
    odom_topic= "true_odom"
    velocity_topic= "cmd_vel"
    link_name= 'robot_vacuum_cleaner::base_link'    
    publish_rate= 10
    link_state_transformer= Link_State_Transformer(odom_topic= odom_topic, velocity_topic= velocity_topic,
                            link_states_topic= link_states_topic, link_name= link_name, publish_rate= publish_rate)
    link_state_transformer.execute()



if __name__=="__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass