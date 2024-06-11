import roslib
import rospy
import math
import tf
import time
import numpy as np
import geometry_msgs.msg
import actionlib
import rospkg
import time
import argparse

rospy.init_node('node_controller')

pose_pub = rospy.Publisher(
    '/equilibrium_pose', 
    geometry_msgs.msg.PoseStamped, 
    queue_size=1)

rate = rospy.Rate(40)

pos_x = 0.3896700014366409, 
pos_y = -0.24520208434400215
pos_z = 0.04351502500163676

quat_w = 0.44855754646920937
quat_x = -0.5532593365102323
quat_y = -0.47663443469711675
quat_z = -0.5152807104930058

robot_pose_msg = geometry_msgs.msg.PoseStamped()
robot_pose_msg.pose.position.x = pos_x
robot_pose_msg.pose.position.y = pos_y
robot_pose_msg.pose.position.z = pos_z

robot_pose_msg.pose.orientation.w = quat_w
robot_pose_msg.pose.orientation.x = quat_x
robot_pose_msg.pose.orientation.y = quat_y
robot_pose_msg.pose.orientation.z = quat_z

print("POSITION=", robot_pose_msg.pose.position)
print("ORIENTATION=", robot_pose_msg.pose.orientation)

pose_pub.publish(robot_pose_msg)

print('sent')

rospy.spin()