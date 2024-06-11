#!/usr/bin/env python  
import os
import rospy
import tf
import time
import numpy as np
import pandas as pd
import geometry_msgs.msg
import time
import argparse
import sys
import matplotlib.pyplot as plt
import json
import machinevisiontoolbox as mvt
from datetime import datetime
import matplotlib
matplotlib.use('TKAgg')
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_srvs.srv import Empty, EmptyRequest
import cv2
import os

from darknet_ros_msgs.msg import BoundingBox
import utils

sys.path.append("/home/sayantan/repositories/panda-vision-imitation")
from panda_vision_imitation.logging import write_dict, write_np_dict, custom_logging_setup_simple
from panda_vision_imitation.data import from_tangent_plane
from panda_vision_imitation.utils import assert_square

# For testing with a fixed goal at center of image
from test_center_goal import get_center_goal

# How to run: 
# 1. On robot computer:
# roscore
# roslaunch franka_example_controllers cartesian_impedance_example_controller.launch robot_ip:=192.168.64.147 load_gripper:=true
# python franka_gripper_open_server.py
#
# On camera computer: 
# 2. roslaunch realsense2_camera rs_camera.launch color_width:=640 color_height:=480 color_fps:=30
# 3. roslaunch darknet_ros darknet_ros.launch
# 4. python ros_scripts/bbox_tracker.py --tracked_obj_class mouse --prob_thresh 0.4 # <<< CHANGE OBJECT AND THRESHOLD AS NEEDED!
# 5. python ros_scripts/bbox_image_pub.py --test_center_goal
# 6. rosrun image_view image_view image:=/img_bbox_tracker
# 7. MOUSE: python ros_scripts/robot_YOLO.py --max_steps 900 --gain_lambda 0.5 --log_dir results_mouse_20231219 --test_center_goal --deadzone 0 --train_path data_node_mouse/train_data.npy --rob_init --desc pos0
# CUP: python ros_scripts/robot_YOLO.py --max_steps 700 --gain_lambda 0.5 --log_dir results_cup_solo_20240118 --deadzone 0 --train_path data_node_cup_1/train_data.npy --rob_init --desc pos1 --open_gripper
#
# OR to debug with a bagfile
#
# 1. In a terminal start roscore: `roscore`
# 2. In another terminal, play the bagfile: `rosbag play bagfiles/2023-09-22-17-56-03.bag`
#    See bagfiles/bagfile_readme.md for download instructions
# 3. python ros_scripts/robot_YOLO.py --max_steps 900 --gain_lambda 0.5 --log_dir logs_ros --test_center_goal --deadzone 0 --train_path data_node_mouse/train_data.npy --rob_init --desc pos0

def parse():

    parser = argparse.ArgumentParser(description='Control panda using a YOLO controller')
    parser.add_argument('--log_dir', type=str, default='logs', help='Location to save logs in')
    parser.add_argument('--rate', type=int, default=40, help='Frequency of updates')
    # max_steps = 430 for cup, 900 for mouse
    parser.add_argument('--max_steps', type=int, default=20, help='Number of steps')
    parser.add_argument('--gain_lambda', type=float, default=0.5, help='Gain')
    parser.add_argument('--rob_init', action='store_true', help='Use the init pose from training data')    
    parser.set_defaults(rob_init=False) 
    parser.add_argument('--test_center_goal', action='store_true', help='Test with the goal at center')    
    parser.set_defaults(test_center_goal=False) 
    parser.add_argument('--train_path', required=True, help='Path to NODE training data (for fetching goal)')
    parser.add_argument('--deadzone', type=int, default=10, help='Initial steps from training data that are discarded')
    parser.add_argument('--img_width', type=int, default=640, help='Image width')
    parser.add_argument('--img_height', type=int, default=480, help='Image height')
    parser.add_argument('--desc', type=str, required=True, help='Description of experiment')
    parser.add_argument('--open_gripper', action='store_true', help='Open the gripper at the end')    
    parser.set_defaults(open_gripper=False) 

    args = parser.parse_args()
    return args

# Change to ros logging
def myprint(*args):
    print('----')
    print(*args)

class RobotYOLO():

    def __init__(self):

        # Get commandline args
        self.args = parse()

        # Init target bounding box
        self.bbox = None

        # Initialize ROS node
        rospy.init_node('YOLO_controller')

        # Publisher for moving the robot
        self.pose_pub = rospy.Publisher(
            '/equilibrium_pose', 
            geometry_msgs.msg.PoseStamped, 
            queue_size=1)

        # For getting end-effector pose and orientation
        self.tf_listener = tf.TransformListener()

        # Subscribe for getting the tracked bbox
        rospy.Subscriber(
            'bbox_tracker', 
            BoundingBox, 
            self.tracked_bbox_callback)

        # Init logs
        self.log_init()

    def log_init(self):
        # Initialize logging variables for observations
        self.log_obs_pos = list()  # Observed EE position at each step
        self.log_obs_quat_wxyz = list()  # Observed EE orientation at each step
        self.log_bbox_errs = list()  # bbox error at each step
        self.log_bbox = list()  # list of dicts {xmin, xmax, ymin, ymax} from YOLO
        self.log_fin_img = None  # Image at final step showing curr and goal bbox
        self.log_curr_bbox_err = None  # For logging the bbox error in the control loop
        self.log_list = list()

    def log_observations(self, step:int):

        log_msg = ''
        ee_pos_xyz, ee_quat_xyzw = None, None
        # Log the EE pose
        for i in range(1):
            try:
                
                (ee_pos_xyz, ee_quat_xyzw) = self.tf_listener.lookupTransform('/panda_link0', '/panda_EE', rospy.Time(0))                
                # Convert quat to wxyz format
                order = [3, 0, 1, 2]
                ee_quat_wxyz = [ee_quat_xyzw[i] for i in order]
            except Exception as e:
                log_msg=f'Exception in pose: {e}'
                print(log_msg)

        # Log the YOLO bbox
        yolo_bbox = dict(xmin=self.bbox.xmin,
                         xmax=self.bbox.xmax,
                         ymin=self.bbox.ymin,
                         ymax=self.bbox.ymax)

        # Log the YOLO bbox error
        bbox_err_norm = self.log_curr_bbox_err

        log_step = dict(step_id=step,
                        ee_pos_x=ee_pos_xyz[0],
                        ee_pos_y=ee_pos_xyz[1],
                        ee_pos_z=ee_pos_xyz[2],
                        ee_quat_w=ee_quat_wxyz[0],
                        ee_quat_x=ee_quat_wxyz[1],
                        ee_quat_y=ee_quat_wxyz[2],
                        ee_quat_z=ee_quat_wxyz[3],
                        yolo_bbox_xmin=yolo_bbox['xmin'],
                        yolo_bbox_xmax=yolo_bbox['xmax'],
                        yolo_bbox_ymin=yolo_bbox['ymin'],
                        yolo_bbox_ymax=yolo_bbox['ymax'],
                        bbox_err_norm=bbox_err_norm,
                        log_msg=log_msg,
                        )
        
        self.log_list.append(log_step)

    def log_store(self, log_dir, desc):
        # Store the log results on the disk
        df = pd.DataFrame(self.log_list)

        # Log name
        script_path = os.path.abspath(__file__)
        script_directory, script_name = os.path.split(script_path)

        script_name = script_name.replace(".py", "")
        log_name = f'{script_name}_{desc}_{datetime.today().strftime("%Y%m%d_%H%M%S")}.csv'

        # Save to CSV
        df.to_csv(os.path.join(log_dir, log_name), index=False)

    def open_gripper(self):
        rospy.wait_for_service('franka_gripper_open')

        try:
            franka_gripper_open = rospy.ServiceProxy('franka_gripper_open', Empty)
            response = franka_gripper_open()
            print("Franka gripper open service called successfully!")
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")

    def tracked_bbox_callback(self, bbox):
        self.bbox = bbox

    def skew_mat(self, p):
        return np.array([
            [0, -p[2], p[1]],
            [p[2], 0, -p[0]],
            [-p[1], p[0], 0]
        ])
    
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
        myprint(f'Position={robot_pose_msg.pose.position}')
        myprint(f'Orientation={robot_pose_msg.pose.orientation}')

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
        """
        Transforms rotation matrix and translation vector
        to a 4x4 transformation matrix
        """

        if rotation_matrix.shape != (3, 3) or translation_vector.shape != (3, 1):
            raise ValueError('Rotation matrix must be 3x3, translation vector must be 3x1')

        # Create a 4x4 identity matrix
        transformation_matrix = np.eye(4)

        # Copy the rotation matrix into the top-left 3x3 block
        transformation_matrix[:3, :3] = rotation_matrix

        # Copy the translation vector into the rightmost column
        transformation_matrix[:3, 3] = translation_vector.flatten()

        return transformation_matrix
    
    def quaternion_to_matrix(self, q):
        """
        Convert a quaternion to a 4x4 transformation matrix.
        """
        x, y, z, w = q
        return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y, 0],
            [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x, 0],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2, 0],
            [0, 0, 0, 1]
        ])

    def create_transformation_matrix(self, translation, quaternion):
        """
        Create a 4x4 transformation matrix from a translation vector and a quaternion vector.
        """
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = translation

        rotation_matrix = self.quaternion_to_matrix(quaternion)

        return np.dot(translation_matrix, rotation_matrix)

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
        """
        bbox_4_np has [xmin,ymin,xmax,ymax]
        This is converted to bbox_8_np (anticlockwise), which has
        [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]
        """

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
        TRAIN_DATA = self.args.train_path
        DEMO_ID = 0
        DEADZONE = self.args.deadzone

        # Image size - must match size of images published by camera
        IMAGE_WIDTH = self.args.img_width
        IMAGE_HEIGHT = self.args.img_height

        # Dist between EE and top of cup measured manually 
        DEPTH_GOAL_SCALAR = 0.17 
        
        # Sampling frequency
        dt = float(1.0/self.args.rate)
        
        # Set the update rate
        rate = rospy.Rate(self.args.rate)

        # Fetch init pose from training data
        train_data = np.load(TRAIN_DATA)[DEMO_ID, DEADZONE:, :]

        # Goal bbox
        goal_bbox_xyxy4 = train_data[-1,0:4] # bbox is in range [0.0,1.0], shape [4,]

        # If this is a test, set the goal manually
        # in the same format as the training data
        if self.args.test_center_goal:
            goal_bbox_xyxy4 = get_center_goal(image_width=IMAGE_WIDTH,
                                              image_height=IMAGE_HEIGHT,
                                              pixel=False) # bbox is in range [0.0,1.0], shape [4,]

        # Assert goal bbox square
        assert_square(goal_bbox_xyxy4)

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

        # Initializations
        depthGoal= DEPTH_GOAL_SCALAR * np.array([1.0, 1.0, 1.0, 1.0])
        gain = self.args.gain_lambda

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
        cam = mvt.CentralCamera(f=f,pp=pp)

        if self.args.rob_init:
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
        curr_quat_wxyz_first = curr_quat_wxyz.copy()

        # Init
        next_quat_wxyz = curr_quat_wxyz_first.copy()

        pos_for_plot = list()
        quat_for_plot = list()
        errors = list()

        # Control loop
        for i in range(self.args.max_steps):

            myprint(f'Step={i}')

            pos_for_plot.append(curr_pos)
            quat_for_plot.append(next_quat_wxyz)

            # Get YOLO XY pixel coords, shape [8,]
            # Results are already in pixel coordinates
            curr_bbox_xyxy4 = np.array([self.bbox.xmin, self.bbox.ymin,
                                        self.bbox.xmax, self.bbox.ymax])
            assert_square(curr_bbox_xyxy4)

            curr_bbox_xyxy8 = self.bbox_4_to_8(bbox_4_np=curr_bbox_xyxy4,
                                               to_pixel=False)
            myprint(f'curr_bbox_xyxy8={curr_bbox_xyxy8}')

            sCurr = curr_bbox_xyxy8 

            # Compute visual error
            error = np.reshape(sCurr - sGoal, (8,1), 'F')
            myprint(f'error={error}')

            # Compute the norm of the visual error
            normE = np.linalg.norm(error)
            self.log_curr_bbox_err = normE
            myprint(f'normE={normE}')

            errors.append(normE)
            
            # Compute the interaction matrix and the null projector for the classic error
            # visual jacobian - Ls is 6x8 
            Ls = cam.visjac_p(np.reshape(sCurr, (2,4), 'F'), depthGoal) 
            #myprint(f'Ls={Ls}')

            # Null space projector
            Ps = np.eye(Ls.shape[1]) - np.linalg.pinv(Ls) @ Ls

            # Compute interaction matrix and null projector for the error norm
            # LvPinv is J in the equation
            Lv, LvPinv, Pv = utils.computeLvPinv(error, Ls) 
            #myprint(f'Lv={Lv}')
            #myprint(f'LvPinv={LvPinv}')

            # Compute the variable for the switch (on the error)
            alpha = utils.switchingRule(normE, e0=10, e1=20) # TODO tune later if needed

            task1_1 = LvPinv * (-gain * normE)
            task1_2 = - gain * np.linalg.pinv(Ls) @ error
            task1 = alpha * task1_1 + (1-alpha) * task1_2
        
            # Camera velocity command
            vCam = task1.flatten()
            myprint(f'vCam (calculated)={vCam}')

            #vCam = np.array([0.0, 0.0, 0.0, 0.0, 0.05, 0.0])

            # Transform vel from camera to EE
            vEnd = e_V_c @ vCam
            myprint(f'task1_1={task1_1}')
            myprint(f'task1_2={task1_2}')

            #myprint(f'vCam={vCam}')
            #myprint(f'vEnd={vEnd}')
            #myprint(f'e_V_c={e_V_c}')

            #vEnd_flat = vEnd.flatten()

            # Get gripper pose in base frame
            # rot is in xyzw
            trans,rot = None, None
            try:	
                (trans,rot) = self.tf_listener.lookupTransform('/panda_link0', '/panda_EE', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                myprint(f'Could not get current pose')
                continue

            # Convert to matrices
            b_T_ee = self.create_transformation_matrix(translation=trans,
                                                       quaternion=rot)

            # Transform vel from EE to base
            b_V_e = self.get_vel_twist(ee_T_c=b_T_ee)
            vBase = b_V_e @ vEnd

            #myprint(f'trans={trans}')
            #myprint(f'rot={rot}')
            #myprint(f'b_T_ee={b_T_ee}')
            #myprint(f'b_V_e={b_V_e}')
            #myprint(f'vBase={vBase}')

            # Manually compute next pose using linear velocity
            next_pos = curr_pos + dt*vBase[0:3]

            # Integrate EE angular velocity using exp map
            # to get next quaternion
            del_quat_wxyz = from_tangent_plane(
                #q_goal=goal_quat_wxyz, 
                q_goal=next_quat_wxyz, 
                r=(dt/2.0)*vBase[3:6])
            #next_quat_wxyz = utils.quaternion_multiply(q1_wxyz=del_quat_wxyz,
            #                                           q2_wxyz=curr_quat_wxyz)
            next_quat_wxyz = del_quat_wxyz
            #next_quat_wxyz = curr_quat_wxyz # Do not update quat (for testing) # TODO remove
            #myprint(f'norm of next_quat_wxyz={np.linalg.norm(next_quat_wxyz)}')

            #myprint(f'del_quat_wxyz={del_quat_wxyz}')
            #myprint(f'curr_quat_wxyz_first={curr_quat_wxyz_first}')

            #myprint(f'curr_pos={curr_pos}')
            #myprint(f'next_pos={next_pos}')
            #myprint(f'next_quat_wxyz={next_quat_wxyz}')

            # Send to robot TODO
            # sleep is taken care by move function

            # Moving robot to next pose
            myprint('Moving robot to next pose')
            self.move_robot(
                pos_xyz=next_pos, 
                quat_wxyz=next_quat_wxyz, 
                rate=rate, 
                init=False)

            self.log_observations(step=i)

            # Set next to curr
            curr_pos = next_pos
            #curr_quat_wxyz = next_quat_wxyz

        myprint('Loop done')

        pos_for_plot = np.array(pos_for_plot)
        quat_for_plot = np.array(quat_for_plot)
        errors = np.array(errors)

        if self.args.open_gripper:
            self.open_gripper()

        # Save logs
        self.log_store(log_dir=self.args.log_dir, desc=self.args.desc)

r = RobotYOLO()
r.main()