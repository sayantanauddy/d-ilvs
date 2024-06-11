#!/usr/bin/env python  
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
import torch
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import os
from geometry_msgs.msg import PoseStamped

## Steps:
# 0: Go to root of repo
# 1. launch Franka controller: roslaunch franka_example_controllers cartesian_impedance_example_controller.launch robot_ip:=192.168.64.147 load_gripper:=true
# 2. launch Realsense camera: roslaunch realsense2_camera rs_camera.launch
# 3. launch aruco_ros: roslaunch aruco_ros single.launch 
# 4. view aruco: rosrun image_view image_view image:=/aruco_single/result (check z does not flip)
# 5. Repeat n times:
    # 5.0 Place arm in different poses (high above table)
    # 5.1 View aruko marker image (make sure z-axis does not flip between observations)
    # 5.2 run this script: python ros_scripts/calibrate_record.py (each obs is saved in calibration_data) 
# 6. Run python ros_scripts/calibrate_process.py to process all observations ()

class Calibrate():

    def __init__(self):

        # Initialize ROS node
        rospy.init_node('hand_eye')

        # Subscribe to the aruco topic to get the 
        # target pose in the camera frame
        rospy.Subscriber('/aruco_single/pose', 
                        PoseStamped, 
                        self.aruco_callback)

        # For getting end-effector pose and orientation
        self.listener = tf.TransformListener()

        self.R_target2cam = None
        self.t_target2cam = None
        self.R_gripper2base = None
        self.t_gripper2base = None

    def aruco_callback(self, target_pose_in_camera_frame):
        
        # Fetch the latest target pose in the camera frame
        self.target_pose_in_camera_frame = target_pose_in_camera_frame

        # Construct R_target2cam
        # Rotation part extracted from the homogeneous matrix that transforms 
        # a point expressed in the target frame to the camera frame (cTt). 
        # This is a (3x3) rotation matrix, for all the transformations 
        # from calibration target frame to camera frame.
        self.R_target2cam = R.from_quat([
            self.target_pose_in_camera_frame.pose.orientation.x,
            self.target_pose_in_camera_frame.pose.orientation.y,
            self.target_pose_in_camera_frame.pose.orientation.z,
            self.target_pose_in_camera_frame.pose.orientation.w]).as_matrix()
        
        # Construct t_target2cam
        # Translation part extracted from the homogeneous matrix that transforms 
        # a point expressed in the target frame to the camera frame (cTt). 
        # This is a vector that contains the (3x1) translation vectors for all 
        # the transformations from calibration target frame to camera frame.
        self.t_target2cam = np.array([
            [self.target_pose_in_camera_frame.pose.position.x],
            [self.target_pose_in_camera_frame.pose.position.y],
            [self.target_pose_in_camera_frame.pose.position.z]])

    def record_gripper(self):

        trans,rot = None, None
        for i in range(5):
            try:	
                # Get gripper pose
                (trans,rot) = self.listener.lookupTransform('/panda_link0', '/panda_EE', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                time.sleep(1)
                continue
        
        if trans is None or rot is None:
            print('Robot EE pose not found')
            exit(0)

        ee_pose = dict(
            ee_pos_x = trans[0],
            ee_pos_y = trans[1],
            ee_pos_z = trans[2],
            ee_quat_w = rot[3],
            ee_quat_x = rot[0],
            ee_quat_y = rot[1],
            ee_quat_z = rot[2]
            )
        
        # Construct rotation matrix R_gripper2base
        # Rotation part extracted from the homogeneous matrix that transforms 
        # a point expressed in the gripper frame to the robot base frame (bTg). 
        # This is a (3x3) rotation matrix for all the transformations from gripper frame 
        # to robot base frame.
        self.R_gripper2base = R.from_quat([
            ee_pose['ee_quat_x'],
            ee_pose['ee_quat_y'],
            ee_pose['ee_quat_z'],
            ee_pose['ee_quat_w']]).as_matrix()

        # Construct t_gripper2base
        # Translation part extracted from the homogeneous matrix that transforms 
        # a point expressed in the gripper frame to the robot base frame (bTg). 
        # This is a vector that contains the (3x1) translation vectors for all 
        # the transformations from gripper frame to robot base frame.
        self.t_gripper2base = np.array([
            [ee_pose['ee_pos_x']],
            [ee_pose['ee_pos_y']],
            [ee_pose['ee_pos_z']],
        ])

    def show(self):

        print('R_gripper2base:', self.R_gripper2base)
        print('t_gripper2base:', self.t_gripper2base)
        print('R_target2cam:', self.R_target2cam)
        print('t_target2cam:', self.t_target2cam)
        

he = Calibrate()
he.record_gripper()
data = dict(
    R_gripper2base=he.R_gripper2base,
    t_gripper2base=he.t_gripper2base,
    R_target2cam=he.R_target2cam,
    t_target2cam=he.t_target2cam,
)
he.show()

save_dir = 'calibration_data'
save_file = f'calibration_{datetime.today().strftime("%Y%m%d_%H%M%S")}.npz'
np.savez(os.path.join(save_dir, save_file), **data)
