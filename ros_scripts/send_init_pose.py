#!/usr/bin/env python  
import roslib
import rospy
import math
import tf
import time
import numpy
import geometry_msgs.msg
import rospkg
import time
import argparse

# roslaunch franka_example_controllers cartesian_impedance_example_controller.launch robot_ip:=192.168.64.147 load_gripper:=true

if __name__ == '__main__':


    time.sleep(5)
    rospy.init_node('send_grasp_pose')

    pose_pub = rospy.Publisher('/equilibrium_pose', geometry_msgs.msg.PoseStamped, queue_size=1)
    robot_pose_msg = geometry_msgs.msg.PoseStamped()
    
    rate = rospy.Rate(40.0) # 30 Hz
    
    # Send trajectory
    for i in range(20):

        # Exit if ctrl+c is pressed
        if rospy.is_shutdown():
            break 

        # Position
        pos_x = float(0.3896700014366409)
        pos_y = float(-0.24520208434400215)
        pos_z = float(0.04351502500163676)

        quat_w = float(0.44855754646920937)
        quat_x = float(-0.5532593365102323)
        quat_y = float(-0.47663443469711675)
        quat_z = float(-0.5152807104930058)

        robot_pose_msg.pose.position.x = pos_x
        robot_pose_msg.pose.position.y = pos_y
        robot_pose_msg.pose.position.z = pos_z

        # Orientation (as unit quaternion)
        robot_pose_msg.pose.orientation.w = quat_w
        robot_pose_msg.pose.orientation.x = quat_x
        robot_pose_msg.pose.orientation.y = quat_y
        robot_pose_msg.pose.orientation.z = quat_z

        print("POSITION=", robot_pose_msg.pose.position)
        print("ORIENTATION=", robot_pose_msg.pose.orientation)

        pose_pub.publish(robot_pose_msg)

        rate.sleep()

    time.sleep(1.0)
