#!/usr/bin/env python  
import rospy
import tf
import time
import numpy as np
import geometry_msgs.msg
import time
import argparse
import sys
import matplotlib.pyplot as plt
import json
import machinevisiontoolbox as mvt
from datetime import datetime

from darknet_ros_msgs.msg import BoundingBox
import utils

sys.path.append("/home/sayantan/repositories/panda-vision-imitation")
from panda_vision_imitation.data import from_tangent_plane

# TODO: Implement logging to file with ROS

# Provide hardcoded velocities in the camera frame and move EE
# How to run: 
# 1. On robot computer:
# roslaunch franka_example_controllers cartesian_impedance_example_controller.launch robot_ip:=192.168.64.147 load_gripper:=true
# 
# On camera computer: 
# 2. python ros_scripts/test_cam_vel.py

def parse():

    parser = argparse.ArgumentParser(description='Control panda using a YOLO controller')
    parser.add_argument('--rate', type=int, default=40, help='Frequency of updates')
    parser.add_argument('--max_steps', type=int, default=30, help='Number of steps')
    parser.add_argument('--rob_init', action='store_true', help='Use the init pose from training data')    
    parser.set_defaults(rob_init=False) 

    args = parser.parse_args()
    return args

# Change to ros logging
def myprint(*args):
    print('----')
    print(*args)

class testVel():

    def __init__(self, vCam: np.array):

        self.bbox = None

        # Hardcoded camera velocity of shape [6,]
        self.vCam = vCam

        #### ROS setup

        # Initialize ROS node
        rospy.init_node('test_cam_vel')

        # Publish to the topic for moving the robot
        self.pose_pub = rospy.Publisher(
            '/equilibrium_pose', 
            geometry_msgs.msg.PoseStamped, 
            queue_size=1)

        # For getting end-effector pose and orientation
        self.tf_listener = tf.TransformListener()

    def skew_mat(self, p):
        return np.array([
            [0, -p[2], p[1]],
            [p[2], 0, -p[0]],
            [-p[1], p[0], 0]
        ])
    
    def transform_cam2ee(self, vel_cam):
        # Linear and angular velocity in EE frame
        vel_ee = self.e_V_c @ vel_cam
        return vel_ee

    def move_robot(self, pos_xyz, quat_wxyz, rate, init=False, init_wait=5):

        assert pos_xyz.shape == (3,), f'Shape of pos_xyz: {pos_xyz.shape}'
        assert quat_wxyz.shape == (4,), f'Shape of quat_wxyz: {quat_wxyz.shape}'
        assert np.allclose(np.linalg.norm(quat_wxyz), 1.0)

        # Position should be in m
        assert np.max(pos_xyz) < 2.0

        # Set init pose in message
        robot_pose_msg = geometry_msgs.msg.PoseStamped()
        robot_pose_msg.pose.position.x = pos_xyz[0]
        robot_pose_msg.pose.position.y = pos_xyz[1]
        robot_pose_msg.pose.position.z = pos_xyz[2]
        
        robot_pose_msg.pose.orientation.w = quat_wxyz[0]
        robot_pose_msg.pose.orientation.x = quat_wxyz[1]
        robot_pose_msg.pose.orientation.y = quat_wxyz[2]
        robot_pose_msg.pose.orientation.z = quat_wxyz[3]

        #myprint(f'Moving robot to the following pose (init={init})')
        #myprint('Position=', robot_pose_msg.pose.position)
        #myprint('Orientation=', robot_pose_msg.pose.orientation)

        if init:
            myprint('Setting robot in init pose ...')      
            # Give some time to the robot to go to initial pos
            for i in range(20):
                # Exit if ctrl+c is pressed
                if rospy.is_shutdown():
                    break 
                # Send init pose to robot
                self.pose_pub.publish(robot_pose_msg)
                rate.sleep()
            time.sleep(init_wait)
        else:
            self.pose_pub.publish(robot_pose_msg)
            rate.sleep()

    def homogeneous_transform(self, rotation_matrix, translation_vector):
        '''
        Transforms rotation matrix and translation vector
        to a 4x4 transformation matrix
        '''

        if rotation_matrix.shape != (3, 3) or translation_vector.shape != (3, 1):
            raise ValueError('Rotation matrix must be 3x3, translation vector must be 3x1')

        # Create a 4x4 identity matrix
        transformation_matrix = np.eye(4)

        # Copy the rotation matrix into the top-left 3x3 block
        transformation_matrix[:3, :3] = rotation_matrix

        # Copy the translation vector into the rightmost column
        transformation_matrix[:3, 3] = translation_vector.flatten()

        return transformation_matrix
    
    def read_json_file(self, file_path):
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        return data

    def load_eeTc(self, R_cam2gripper_PATH, t_cam2gripper_PATH):

        # The calibration process needs to be run before this script
        # Check ros_scripts/calibration_record.py
        # Load
        R_cam2gripper = np.load(R_cam2gripper_PATH)
        t_cam2gripper = np.load(t_cam2gripper_PATH)
        ee_T_c = self.homogeneous_transform(rotation_matrix=R_cam2gripper,
                                            translation_vector=t_cam2gripper)
        return ee_T_c
    
    def get_vel_twist(self, ee_T_c):
        # Velocity twist transformations
        e_p_c = ee_T_c[0:3,3]
        e_R_c = ee_T_c[0:3,0:3]
        e_V_c = np.zeros((6, 6))
        e_V_c[0:3, 0:3] = e_R_c
        e_V_c[3:6, 3:6] = e_R_c
        e_V_c[0:3, 3:6] = self.skew_mat(e_p_c) @ e_R_c
        return e_V_c
    
    def main(self):

        # Robot and camera settings saved on disk
        R_cam2gripper_PATH = 'calibration_data/R_cam2gripper.npy'
        t_cam2gripper_PATH = 'calibration_data/t_cam2gripper.npy'
        INTRINSIC_PATH = 'calibration_data/realsense_intrinsic.json'
        
        # Training data is used to fetch goal bbox
        TRAIN_DATA = 'data_node/train_data.npy'
        DEMO_ID = 0
        DEADZONE = 70

        # Dist between EE and top of cup measured manually 
        DEPTH_GOAL_SCALAR = 0.17 
        
        NUM_TRAIN_STEPS = 430

        # Get args
        args = parse()

        dt = float(1.0/args.rate)
        
        # Set the update rate
        rate = rospy.Rate(args.rate)

        # Fetch init pose from training data
        train_data = np.load(TRAIN_DATA)[DEMO_ID, DEADZONE:, :]
        assert train_data.shape == (NUM_TRAIN_STEPS, 11)

        # Fetch the goal quat (for rot vec calculation later)
        goal_quat_wxyz = train_data[-1,7:11]
                   
        # Get camera parameters
        #intrinsic, ee_T_c = utils.get_camera_parameters()
        intrinsic = self.read_json_file(INTRINSIC_PATH)
        ee_T_c = self.load_eeTc(R_cam2gripper_PATH=R_cam2gripper_PATH,
                                t_cam2gripper_PATH=t_cam2gripper_PATH)
        
        e_V_c = self.get_vel_twist(ee_T_c=ee_T_c)

        myprint(f'intrinsic={intrinsic}')
        myprint(f'ee_T_c={ee_T_c} shape={ee_T_c.shape}')
        myprint(f'e_V_c={e_V_c} shape={e_V_c.shape}')

        # Initializations
        depthGoal= DEPTH_GOAL_SCALAR * np.array([1.0, 1.0, 1.0, 1.0])
        myprint(f'depthGoal={depthGoal}')

        vCam = np.zeros((6,1)) # velocity of camera

        # Define the camera 
        f = np.array([intrinsic['fx'],intrinsic['fy']])
        pp = np.array([intrinsic['cx'],intrinsic['cy']])
        cam = mvt.CentralCamera(f=f*1e-5,pp=pp)

        myprint('Not changing init pose')

        # Get the current pose of the robot
        # rot is in xyzw
        trans,rot = None, None
        for i in range(5):
            try:	
                # Get gripper pose
                (trans,rot) = self.tf_listener.lookupTransform('/panda_link0', '/panda_EE', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                time.sleep(1)
                continue
        if trans is None or rot is None:
            myprint('Could not get current pose of robot')
            exit(1)

        curr_pos = trans
        curr_quat_wxyz = np.array([rot[3], rot[0], rot[1], rot[2]])
        myprint(f'curr_pos={curr_pos}')
        myprint(f'curr_quat_wxyz={curr_quat_wxyz}')

        pos_for_plot = list()
        quat_for_plot = list()
        errors = list()

        # Control loop
        for i in range(args.max_steps):

            pos_for_plot.append(curr_pos)
            quat_for_plot.append(curr_quat_wxyz)
        
            # Camera velocity command
            vCam = self.vCam

            # Transform vel from camera to EE
            vEnd = e_V_c @ vCam

            vEnd_flat = vEnd.flatten()

            # Manually compute next pose using linear velocity
            myprint('dt is hardcoded')
            exit(0)
            next_pos = curr_pos + 0.5*vEnd_flat[0:3]

            # The following works as expected
            #next_pos = curr_pos + np.array([0.0, 0.0, 0.005])
            # This also works
            #next_pos = curr_pos + dt*np.array([0.0, 0.0, 0.2])

            # Integrate EE angular velocity using exp map
            # to get next quaternion
            del_quat_wxyz = from_tangent_plane(
                q_goal=goal_quat_wxyz, 
                r=(dt/2.0)*vEnd_flat[3:6])
            next_quat_wxyz = utils.quaternion_multiply(q1_wxyz=del_quat_wxyz,
                                                       q2_wxyz=curr_quat_wxyz)
            next_quat_wxyz = curr_quat_wxyz # Do not update quat (for testing) # TODO remove

            myprint(f'Step={i}')
            myprint(f'vEnd={vEnd}')
            myprint(f'vCam={vCam}')
            myprint(f'vEnd_flat[0:3]={vEnd_flat[0:3]}')
            myprint(f'dt={dt}')
            myprint(f'dt*vEnd_flat[0:3]={dt*vEnd_flat[0:3]}')
            myprint(f'curr_pos={curr_pos}')
            myprint(f'curr_quat_wxyz={curr_quat_wxyz}')
            myprint(f'next_pos={next_pos}')
            myprint(f'next_quat_wxyz={next_quat_wxyz}')
            myprint(f'Delta position: {curr_pos - next_pos}')
            myprint(f'Delta quat: {next_quat_wxyz - curr_quat_wxyz}')

            # Moving robot to next pose
            # sleep is taken care by move function
            myprint('Moving robot to next pose')
            self.move_robot(
                pos_xyz=next_pos, 
                quat_wxyz=next_quat_wxyz, 
                rate=rate, 
                init=False)

            # Set next to curr
            curr_pos = next_pos
            curr_quat_wxyz = next_quat_wxyz

        myprint('Loop done')

        pos_for_plot = np.array(pos_for_plot)
        quat_for_plot = np.array(quat_for_plot)
        errors = np.array(errors)

vCam = np.array([-0.005, 0.0, 0.0, 0.0, 0.0, 0.0])
tv = testVel(vCam=vCam)
tv.main()