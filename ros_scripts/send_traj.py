#!/usr/bin/env python  

import rospy
import numpy as np
import os
import time
import numpy
import geometry_msgs.msg
#from franka_gripper.msg import HomingAction, HomingActionGoal, GraspAction, GraspGoal
import actionlib
import time
import argparse

def parse():

    parser = argparse.ArgumentParser(description='Replay recorded demo on Franka')
    parser.add_argument('--data_root', type=str, help='Data directory')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse()

    time.sleep(5)
    rospy.init_node('replay_demo')

    # Open Gripper
    #gripper_client = actionlib.SimpleActionClient('/franka_gripper/homing', HomingAction)
    #gripper_client.wait_for_server()
    #gripper_goal = HomingActionGoal()
    #gripper_client.send_goal(gripper_goal)
    #gripper_client.wait_for_result()

    #grasp_goal = GraspGoal()
    #grasp_goal.width = 0.02
    #grasp_goal.epsilon.inner = 0.015
    #grasp_goal.epsilon.outer = 0.03
    #grasp_goal.speed = 0.1
    #grasp_goal.force = 40.0
    #gripper_client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)

    pose_pub = rospy.Publisher('/equilibrium_pose', geometry_msgs.msg.PoseStamped, queue_size=1)
    robot_pose_msg = geometry_msgs.msg.PoseStamped()
    
    rate = rospy.Rate(40.0)
    
    # Load desired trajectory
    rob_traj = np.loadtxt(os.path.join(args.data_root, 'demo.txt'))

    # Send trajectory point by point
    for pose_ in rob_traj:
        # Exit if ctrl+c is pressed
        if rospy.is_shutdown():
            break 
        # Position
        robot_pose_msg.pose.position.x = pose_[0]
        robot_pose_msg.pose.position.y = pose_[1]
        robot_pose_msg.pose.position.z = pose_[2]
        # Orientation (as unit quaternion)
        q = pose_[3:7] / numpy.linalg.norm(pose_[3:7]) # Normalize to be sure

        robot_pose_msg.pose.orientation.w = pose_[3]
        robot_pose_msg.pose.orientation.x = pose_[4]
        robot_pose_msg.pose.orientation.y = pose_[5]
        robot_pose_msg.pose.orientation.z = pose_[6]

        print("POSITION=")
        print(robot_pose_msg.pose.position)
        #print("ORIENTATION=")
        #print(q)

        pose_pub.publish(robot_pose_msg)
        print(robot_pose_msg)
        rate.sleep()

    time.sleep(1.0)


    #gripper_client = actionlib.SimpleActionClient('/franka_gripper/homing', HomingAction)
    #gripper_client.wait_for_server()
    #gripper_goal = HomingActionGoal()
    #gripper_client.send_goal(gripper_goal)
    #gripper_client.wait_for_result()
