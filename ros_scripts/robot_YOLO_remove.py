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

# How to run: 
# 1. On robot computer:
# roslaunch franka_example_controllers cartesian_impedance_example_controller.launch robot_ip:=192.168.64.147 load_gripper:=true
# 
# On camera computer: 
# 2. roslaunch realsense2_camera rs_camera.launch color_width:=640 color_height:=480 color_fps:=30
# 3. roslaunch darknet_ros darknet_ros.launch
# 4. python ros_scripts/bbox_tracker.py
# 5. python ros_scripts/robot_YOLO.py

def parse():

    parser = argparse.ArgumentParser(description='Control panda using a YOLO controller')
    parser.add_argument('--rate', type=int, default=40, help='Frequency of updates')
    parser.add_argument('--max_steps', type=int, default=20, help='Number of steps')
    parser.add_argument('--rob_init', action='store_true', help='Use the init pose from training data')    
    parser.set_defaults(rob_init=False) 

    args = parser.parse_args()
    return args

# Change to ros logging
def myprint(*args):
    print('----')
    print(*args)

class robotYOLO():

    def __init__(self):

        self.bbox = None

        #### ROS setup

        # Initialize ROS node
        rospy.init_node('YOLO_controller')

        # Publish to the topic for moving the robot
        self.pose_pub = rospy.Publisher(
            '/equilibrium_pose', 
            geometry_msgs.msg.PoseStamped, 
            queue_size=1)

        # For getting end-effector pose and orientation
        self.tf_listener = tf.TransformListener()

        # Subscribe for getting the tracked bbox
        rospy.Subscriber('bbox_tracker', 
                        BoundingBox, 
                        self.tracked_bbox_callback)

    def tracked_bbox_callback(self, bbox):
        self.bbox = bbox

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

        myprint(f'Moving robot to the following pose (init={init})')
        myprint('Position=', robot_pose_msg.pose.position)
        myprint('Orientation=', robot_pose_msg.pose.orientation)

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
    
    def bbox_4_to_8(self, 
                    bbox_4_np, 
                    image_width=None, 
                    image_height=None, 
                    to_pixel=True, 
                    scale=1.0):
        '''
        bbox_4_np has [xmin,ymin,xmax,ymax]
        This is converted to bbox_8_np (anticlockwise), which has
        [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]
        '''
        assert bbox_4_np.shape == (4,)

        xmin = bbox_4_np[0]
        ymin = bbox_4_np[1]
        xmax = bbox_4_np[2]
        ymax = bbox_4_np[3]
        
        # In the file, each coordinate is stored as 
        # a number in the range [0.0,100.0] which 
        # is scaled by the image height and width
        if to_pixel:
            xmin = round((xmin/scale) * image_width)
            ymin = round((ymin/scale) * image_height)
            xmax = round((xmax/scale) * image_width)
            ymax = round((ymax/scale) * image_height)

        bbox_8_np = np.array([xmin, ymin, 
                              xmin, ymax, 
                              xmax, ymax, 
                              xmax, ymin])

        assert bbox_8_np.shape == (8,)

        return bbox_8_np

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
        
        IMAGE_WIDTH = 640
        IMAGE_HEIGHT = 480
        NUM_TRAIN_STEPS = 430
        GAIN_LAMBDA = 0.1

        # Get args
        args = parse()

        dt = float(1.0/args.rate)
        
        # Set the update rate
        rate = rospy.Rate(args.rate)

        # Fetch init pose from training data
        train_data = np.load(TRAIN_DATA)[DEMO_ID, DEADZONE:, :]
        assert train_data.shape == (NUM_TRAIN_STEPS, 11)

        # Goal bbox
        goal_bbox_xyxy4 = train_data[-1,0:4] # bbox is in range [0.0,1.0], shape [4,]

        # Fetch the goal quat (for rot vec calculation later)
        goal_quat_wxyz = train_data[-1,7:11]
        
        # Initial EE pose in training data
        init_pos_xyz_cm = train_data[0,4:7] # position is in cm
        init_quat_wxyz = train_data[0,7:11]

        # Give some time to get first bbox
        while self.bbox is None:
            myprint('waiting for bbox')
            time.sleep(1.0)
            if rospy.is_shutdown():
                return
            
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
        gain = GAIN_LAMBDA

        myprint(f'depthGoal={depthGoal}')
        myprint(f'gain={gain}')

        # Feature value at goal in pixels
        #sGoal = np.array([180.62495498, 235.02389135, 417.40364649, 249.26752041, 409.36712745, 423.50931412, 167.4939879 , 412.31468661])
    
        # bbox pixels in Goal image, load from training data, 
        # transform and convert to shape [8,]
        sGoal = self.bbox_4_to_8(bbox_4_np=goal_bbox_xyxy4,
                                 image_width=IMAGE_WIDTH,
                                 image_height=IMAGE_HEIGHT,
                                 to_pixel=True,
                                 scale=1.0)
        myprint(f'sGoal={sGoal}')

        vCam = np.zeros((6,1)) # velocity of camera

        # Define the camera 
        f = np.array([intrinsic['fx'],intrinsic['fy']])
        pp = np.array([intrinsic['cx'],intrinsic['cy']])
        cam = mvt.CentralCamera(f=f*1e-5,pp=pp)

        if args.rob_init:
            myprint('Setting init pose from training data')
            # Move robot to init pose
            self.move_robot(
                pos_xyz=init_pos_xyz_cm/100.0, 
                quat_wxyz=init_quat_wxyz, 
                rate=rate, 
                init=True,
                init_wait=5)
        else:
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

            myprint(f'Step={i}')

            pos_for_plot.append(curr_pos)
            quat_for_plot.append(curr_quat_wxyz)

            # Get YOLO XY pixel coords, shape [8,]
            # Results are already in pixel coordinates
            curr_bbox_xyxy4 = np.array([self.bbox.xmin, self.bbox.ymin,
                                        self.bbox.xmax, self.bbox.ymax])
            curr_bbox_xyxy8 = self.bbox_4_to_8(bbox_4_np=curr_bbox_xyxy4,
                                               to_pixel=False)
            myprint(f'curr_bbox_xyxy8={curr_bbox_xyxy8}')

            sCurr = curr_bbox_xyxy8 

            # Compute visual error
            error = np.reshape(sCurr - sGoal, (8,1), 'F')
            myprint(error)

            # Compute the norm of the visual error
            normE = np.linalg.norm(error)
            myprint(f'normE={normE}')

            errors.append(normE)
            
            # Compute the interaction matrix and the null projector for the classic error
            # visual jacobian - Ls is 6x8 
            Ls = cam.visjac_p(np.reshape(sCurr, (2,4), 'F'), depthGoal) 

            # Null space projector
            Ps = np.eye(Ls.shape[1]) - np.linalg.pinv(Ls) @ Ls

            # Compute interaction matrix and null projector for the error norm
            # LvPinv is J in the equation
            Lv, LvPinv, Pv = utils.computeLvPinv(error, Ls) 
            myprint(f'Lv={Lv}')
            myprint(f'LvPinv={LvPinv}')

            # Compute the variable for the switch (on the error)
            alpha = utils.switchingRule(normE, e0=25, e1=50) # TODO tune later if needed

            task1_1 = LvPinv * (-gain * normE)
            task1_2 = - gain * np.linalg.pinv(Ls) @ error
            task1 = alpha * task1_1 + (1-alpha) * task1_2
        
            # Camera velocity command
            vCam = task1

            # Transform vel from camera to EE
            vEnd = e_V_c @ vCam

            myprint(f'vEnd={vEnd}')

            vEnd_flat = vEnd.flatten()

            # Manually compute next pose using linear velocity
            next_pos = curr_pos + dt*vEnd_flat[0:3]

            # Integrate EE angular velocity using exp map
            # to get next quaternion
            del_quat_wxyz = from_tangent_plane(
                q_goal=goal_quat_wxyz, 
                r=(dt/2.0)*vEnd_flat[3:6])
            next_quat_wxyz = utils.quaternion_multiply(q1_wxyz=del_quat_wxyz,
                                                       q2_wxyz=curr_quat_wxyz)
            #next_quat_wxyz = curr_quat_wxyz # Do not update quat (for testing) # TODO remove

            myprint(f'next_pos={next_pos}')
            myprint(f'next_quat_wxyz={next_quat_wxyz}')

            # Send to robot TODO
            # sleep is taken care by move function

            # Moving robot to next pose
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

        np.save(f'plots_exp/robot_YOLO_pos_{datetime.today().strftime("%Y%m%d_%H%M%S")}.npy',
                pos_for_plot)
        np.save(f'plots_exp/robot_YOLO_quat_{datetime.today().strftime("%Y%m%d_%H%M%S")}.npy',
                quat_for_plot)
        np.save(f'plots_exp/robot_YOLO_error_{datetime.today().strftime("%Y%m%d_%H%M%S")}.npy',
                errors)


r = robotYOLO()
r.main()