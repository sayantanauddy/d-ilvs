#!/usr/bin/env python  
import roslib
import rospy
import math
import tf
import time
import numpy as np
import pandas as pd
import geometry_msgs.msg
import actionlib
import rospkg
import time
import argparse
import torch
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_srvs.srv import Empty, EmptyRequest
import cv2
import os

from darknet_ros_msgs.msg import BoundingBox

sys.path.append("/home/sayantan/repositories/panda-vision-imitation")
from panda_vision_imitation.model import NODE
from panda_vision_imitation.data import to_tangent_plane, from_tangent_plane
from panda_vision_imitation.utils import assert_square

# How to run: 
#
# 1. On robot computer:
# roscore
# roslaunch franka_example_controllers cartesian_impedance_example_controller.launch robot_ip:=192.168.64.147 load_gripper:=true
# python franka_gripper_open_server.py (in ros_ws/src/my_franka_gripper)
#
# On camera computer: 
# 2. roslaunch realsense2_camera rs_camera.launch color_width:=640 color_height:=480 color_fps:=30
# 3. roslaunch darknet_ros darknet_ros.launch
# 4. python ros_scripts/bbox_tracker.py --tracked_obj_class cup --prob_thresh 0.4  # <<< CHANGE OBJECT AND THRESHOLD AS NEEDED!
# 5. python ros_scripts/bbox_image_pub.py --test_center_goal
# 6. rosrun image_view image_view image:=/img_bbox_tracker
# MOUSE: python ros_scripts/robot_NODE.py --log_dir results_mouse_20231219 --max_steps 900 --train_steps 500 --deadzone 0 --train_path data_node_mouse/train_data.npy --node_path logs/node_mouse/231212_150130_seed100/models/node.pt --desc pos0
# CUP: python ros_scripts/robot_NODE.py --log_dir results_cup_solo_20240118 --max_steps 700 --train_steps 500 --deadzone 0 --train_path data_node_cup_1/train_data.npy --node_path logs/node_cup_1/240117_182213_seed100/models/node.pt --desc pos1 --open_gripper

# # OR to debug with a bagfile
#
# 1. In a terminal start roscore: `roscore`
# 2. In another terminal, play the bagfile: `rosbag play bagfiles/2023-09-22-17-56-03.bag`
#    See bagfiles/bagfile_readme.md for download instructions
# 3. python ros_scripts/robot_NODE.py --log_dir logs_ros --max_steps 900 --train_steps 500 --deadzone 0 --train_path data_node_mouse/train_data.npy --node_path logs/node_mouse/231212_150130_seed100/models/node.pt --desc pos4

def parse():

    parser = argparse.ArgumentParser(description='Control panda using a Neural ODE (NODE)')
    parser.add_argument('--rate', type=int, default=40, help='Frequency of updates')
    # max_steps = 430 for cup, 900 for mouse
    parser.add_argument('--max_steps', type=int, default=500, help='How many steps of execution')
    # train_steps = 430 for cup, 490 for mouse
    parser.add_argument('--train_steps', type=int, default=500, help='How many steps were used in training')
    # deadzone = 70 for cup, 10 for mouse
    parser.add_argument('--deadzone', type=int, default=10, help='Initial steps from training data that are discarded')
    parser.add_argument('--use_curr_bbox', action='store_true', help='Use the current bbox in each step')    
    parser.set_defaults(use_curr_bbox=False) 
    parser.add_argument('--train_path', required=True, help='Path to NODE training data')
    parser.add_argument('--node_path', required=True, help='Path to saved NODE model')
    parser.add_argument('--log_dir', required=True, help='Path to save logs in')
    parser.add_argument('--img_width', type=int, default=640, help='Image width')
    parser.add_argument('--img_height', type=int, default=480, help='Image height')
    parser.add_argument('--desc', type=str, required=True, help='Description of experiment')
    parser.add_argument('--open_gripper', action='store_true', help='Open the gripper at the end')    
    parser.set_defaults(open_gripper=False) 

    args = parser.parse_args()
    return args


class RobotNODE():

    def __init__(self):

        self.bbox = None

        #### ROS setup

        # Initialize ROS node
        rospy.init_node('NODE_controller')

        # Publish to the topic for moving the robot
        self.pose_pub = rospy.Publisher(
            '/equilibrium_pose', 
            geometry_msgs.msg.PoseStamped, 
            queue_size=1)

        # Subscribe for getting the tracked bbox
        rospy.Subscriber('bbox_tracker', 
                        BoundingBox, 
                        self.tracked_bbox_callback)
        
        # Initialize the CvBridge
        self.bridge = CvBridge()

        # Subscribe to the image topic
        rospy.Subscriber('img_bbox_tracker', 
                         Image, 
                         self.image_bbox_callback)
        
        # For getting end-effector pose and orientation
        self.listener = tf.TransformListener()
        
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
                
                (ee_pos_xyz, ee_quat_xyzw) = self.listener.lookupTransform('/panda_link0', '/panda_EE', rospy.Time(0))                
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

    def image_bbox_callback(self, msg):
        # For logging the final observed image
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.log_fin_img = cv_image
        except CvBridgeError as e:
            rospy.logerr(e)
            self.log_fin_img = None
            return

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

        print(f'Moving robot to the following pose (init={init})')
        print('Position=', robot_pose_msg.pose.position)
        print('Orientation=', robot_pose_msg.pose.orientation)

        if init:
            print('Setting robot in init pose ...')      
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

    def get_curr_bbox_for_node(self, scale=100.0):

        # Converts from pixel coordinates to range (0.0,1.0)*scale

        bbox_100_xyxy = np.array([self.bbox.xmin/self.IMAGE_WIDTH, 
                                  self.bbox.ymin/self.IMAGE_HEIGHT, 
                                  self.bbox.xmax/self.IMAGE_WIDTH, 
                                  self.bbox.ymax/self.IMAGE_HEIGHT]) * scale
        
        return bbox_100_xyxy
    
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

        # Set the torch device
        device = torch.device('cuda:0')

        # Get args
        args = parse()

        TRAIN_DATA = args.train_path
        DEMO_ID = 0
        NODE_PATH = args.node_path
        ROTVEC_SCALE = 100.0

        #self.IMAGE_WIDTH = 640
        #self.IMAGE_HEIGHT = 480
        self.IMAGE_WIDTH = args.img_width
        self.IMAGE_HEIGHT = args.img_height
        
        DEADZONE = args.deadzone  # Initial steps from training data that need to be discarded
        NUM_TRAIN_STEPS = args.train_steps  # How many steps were used in training

        # Load trained NODE
        node = torch.load(NODE_PATH)
        
        # Set the update rate
        rate = rospy.Rate(args.rate)
        
        # Give some time to get first bbox
        while self.bbox is None:
            print('waiting for bbox')
            time.sleep(1.0)
            if rospy.is_shutdown():
                return 

        # Fetch init pose from training data
        train_data = np.load(TRAIN_DATA)[DEMO_ID, DEADZONE:, :]
        assert train_data.shape == (NUM_TRAIN_STEPS, 11)

        # Goal bbox
        goal_bbox_xyxy4 = train_data[-1,0:4] # bbox is in range [0.0,1.0], shape [4,]

        # Assert goal bbox square
        assert_square(goal_bbox_xyxy4)

        # bbox pixels in Goal image, load from training data, 
        # transform and convert to shape [8,]
        sGoal = self.bbox_4_to_8(bbox_4_np=goal_bbox_xyxy4,
                                 image_width=self.IMAGE_WIDTH,
                                 image_height=self.IMAGE_HEIGHT,
                                 to_pixel=True,
                                 scale=1.0)
        
        init_bbox_xyxy = train_data[0,0:4] # bbox is in range [0.0,1.0]
        init_pos_xyz = train_data[0,4:7] # position is in cm
        init_quat_wxyz = train_data[0,7:11]

        # Fetch the goal quat (for rot vec calculation later)
        goal_quat_wxyz = train_data[-1,7:11]

        # Check the quaternions770988810724,-0.011787896380281538,410,544,117,251,387.0736364052711,
        assert np.allclose(np.linalg.norm(init_quat_wxyz), 1.0)
        assert np.allclose(np.linalg.norm(goal_quat_wxyz), 1.0)

        # Move robot to init pose
        self.move_robot(
            pos_xyz=init_pos_xyz/100.0, 
            quat_wxyz=init_quat_wxyz, 
            rate=rate, 
            init=True,
            init_wait=5)
        
        # Create t and delta_t for manual integration
        t = np.linspace(0.0, 1.0, NUM_TRAIN_STEPS)
        delta_t = np.ediff1d(t)[0]
        t = torch.from_numpy(t).float().to(device)

        # Create y_start
        pos_cm_xyz = init_pos_xyz
    
        bbox_100_xyxy = self.get_curr_bbox_for_node(scale=100.0)
        rot_vec_scaled_xyz = to_tangent_plane(goal_quat_wxyz, init_quat_wxyz) * ROTVEC_SCALE # scale rot_vec

        y_start = np.concatenate([bbox_100_xyxy, pos_cm_xyz, rot_vec_scaled_xyz])
        y_start = torch.from_numpy(y_start).float().to(device)

        # Use NODE for full integration (for verification)
        y_hat = node(t, y_start.unsqueeze(0))
        y_hat = y_hat.detach().cpu().numpy()[0]

        manual_int_pos = list()
        manual_int_quat = list()

        # Execute the trajectory with manual integration
        for step_id in range(args.max_steps):

            if rospy.is_shutdown():
                return
            
            # Get YOLO XY pixel coords, shape [8,]
            # Results are already in pixel coordinates
            curr_bbox_xyxy4 = np.array([self.bbox.xmin, self.bbox.ymin,
                                        self.bbox.xmax, self.bbox.ymax])
            assert_square(curr_bbox_xyxy4)

            curr_bbox_xyxy8 = self.bbox_4_to_8(bbox_4_np=curr_bbox_xyxy4,
                                               to_pixel=False)
            sCurr = curr_bbox_xyxy8

            # Compute visual error
            error = np.reshape(sCurr - sGoal, (8,1), 'F')

            # Compute the norm of the visual error (only for logging)
            normE = np.linalg.norm(error)
            self.log_curr_bbox_err = normE

            # One step through the target network
            d_y = node.target_network(y_start)

            # Manual integration
            y_next = y_start + delta_t*d_y
            
            # Update the start state
            y_start = y_next

            if args.use_curr_bbox:
                # Fetch the curr bbox from YOLO (each coord is in range [0.0,100.0] for use in the node)
                curr_bbox_100_xyxy = self.get_curr_bbox_for_node(scale=100.0)
                # Set the curr_bbox in y_start
                y_start[0:4] = torch.from_numpy(curr_bbox_100_xyxy).to(device)

            # Get the next position in m
            pos_xyz_cm = y_next[4:7].clone().detach().cpu().numpy()
            pos_xyz_m = pos_xyz_cm/100.0

            # Get the next quaternion
            rot_xyz = y_next[7:].clone().detach().cpu().numpy()
            quat_wxyz = from_tangent_plane(q_goal=goal_quat_wxyz, r=rot_xyz, scale=100.0)
            
            # Send next pose (pos, quat) to robot
            self.move_robot(pos_xyz=pos_xyz_m,
                            quat_wxyz=quat_wxyz,
                            rate=rate,
                            init=False)
            
            self.log_observations(step=step_id)

            manual_int_pos.append(pos_xyz_cm)
            manual_int_quat.append(quat_wxyz)
            
        manual_int_pos = np.array(manual_int_pos)
        manual_int_quat = np.array(manual_int_quat)

        if args.open_gripper:
            self.open_gripper()

        # Save logs
        self.log_store(log_dir=args.log_dir, desc=args.desc)
        
        return

r = RobotNODE()
r.main()