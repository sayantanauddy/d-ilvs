#!/usr/bin/env python  
import rospy
import tf
import time
import datetime 
import numpy as np
import argparse
import os
import json
from cv_bridge import CvBridge, CvBridgeError
import cv2
from copy import deepcopy
from sensor_msgs.msg import JointState, Image
from darknet_ros_msgs.msg import BoundingBox

# How to run: 
# 1. On robot computer:
# Safety should be pressed down (robot LED=white)
# roscore
# roslaunch franka_example_controllers cartesian_impedance_example_controller.launch robot_ip:=192.168.64.147 load_gripper:=true
# 
# On camera computer: 
# 2. roslaunch realsense2_camera rs_camera.launch color_width:=640 color_height:=480 color_fps:=30
# 3. roslaunch darknet_ros darknet_ros.launch
# 4. python ros_scripts/bbox_tracker.py
# 5. (optional) python ros_scripts/bbox_image_pub.py
# 6. (optional) rosrun image_view image_view image:=/img_bbox_tracker
# 7. python ros_scripts/record_demo_images_bbox.py --freq 40 --n_pts 500 --savedir_root /home/sayantan/datasets/panda_vision_traj_v2 --img_width 640 --img_height 480

def parse():

    parser = argparse.ArgumentParser(description='Collect demos on Franka')

    parser.add_argument('--freq', type=float, default=40.0, help='Frequency in Hz')
    parser.add_argument('--n_pts', type=int, default=500, help='How many data points to collect')
    parser.add_argument('--savedir_root', type=str, default='data_img_joint_ee', help='Root dir for saving')
    parser.add_argument('--img_width', type=int, default=640, help='Image width in pixels')
    parser.add_argument('--img_height', type=int, default=480, help='Image height in pixels')

    args = parser.parse_args()
    return args


class FrankaRecorder:

    def __init__(self, args):

        self.freq = args.freq
        self.n_pts = args.n_pts

        self.img_width = args.img_width
        self.img_height = args.img_height

        # Init for holding data from subscribed topics
        self.joint_state = None
        self.color_image = None
        self.bbox = None

        # Current timestamp
        now = datetime.datetime.now()
        timestamp_str = now.strftime('%Y%m%d_%H%M%S')

        # Directory to save demos in
        dir_name = 'demo_' + str(timestamp_str)
        self.save_dir = os.path.join(args.savedir_root,
                                     dir_name)
        self.image_save_dir = os.path.join(self.save_dir, 
                                           'images')

        # File in which demo is saved
        self.robot_filepath = os.path.join(
            self.save_dir, 
            'demo.txt')
        
        # File for saving metadata
        self.metadata_filepath = os.path.join(
            self.save_dir, 
            'metadata.json')

        # Create save directories        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.image_save_dir):
            os.makedirs(self.image_save_dir)

        # Column order in recorded data:
        # EE position (base frame) - 3
        # EE quaternion (base frame) - 4
        # Camera position (base frame) - 3
        # Camera quaternion (base frame) - 4
        # EE position (camera frame) - 3
        # EE quaternion (camera frame) - 4
        # Current bbox (pixels in image plane) - 4
        # Joint positon - 9
        # Joint velocity - 9
        # Joint effort - 9
        self.demonstration = np.ndarray(shape=(args.n_pts, 52))

        ## ROS initialization

        # Initialize this node
        rospy.init_node('save_pose_data')

        # For getting end-effector pose and orientation
        self.listener = tf.TransformListener()
        self.rate = rospy.Rate(args.freq)

        # Callback for getting joint angles
        rospy.Subscriber('joint_states', JointState, self.joint_states_callback)

        # Callback for getting images
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        # Callback for getting images
        rospy.Subscriber('bbox_tracker', BoundingBox, self.tracked_bbox_callback)

        # Instantiate CV bridge to convert Image message to numpy
        self.cv_bridge = CvBridge()

    def tracked_bbox_callback(self, bbox):
        self.bbox = bbox

    def image_callback(self, data):
        self.color_image = data        

    def joint_states_callback(self, data):
        self.joint_state = data

    def write_json_file(self, file_path, data):
        with open(file_path, 'w') as f:
            json.dump(data, f)

    def quat_xyzw_to_wxyz(self, quat_xyzw):
        assert np.array(quat_xyzw).shape == (4,)
        assert np.allclose(np.linalg.norm(quat_xyzw), 1.0)
        x, y, z, w = quat_xyzw
        quat_wxyz = np.array([w,x,y,z])
        return quat_wxyz
    
    def record(self):

        # Give some time to get first bbox
        while self.bbox is None:
            print('waiting for bbox')
            time.sleep(1.0)
            if rospy.is_shutdown():
                return

        # Wait 5 seconds
        wait_time = 5
        for i in range(wait_time):
            print(f'Record in {wait_time-i} seconds ...')
            time.sleep(1.0)
        print('Start recording')

        counter_ = 0
        start = time.time()
        while not rospy.is_shutdown() and counter_ < args.n_pts:
            try:	
                # Get current robot EE pose in base frame
                (trans_base_T_ee, rot_base_T_ee_xyzw) = self.listener.lookupTransform('/panda_link0', '/panda_EE', rospy.Time(0))
                # Convert quaternion from xyzw to wxyz
                rot_base_T_ee_wxyz = self.quat_xyzw_to_wxyz(rot_base_T_ee_xyzw)

                # Get current camera pose in base frame
                (trans_base_T_cam, rot_base_T_cam_xyzw) = self.listener.lookupTransform('/panda_link0', '/camera_link', rospy.Time(0))
                # Convert quaternion from xyzw to wxyz
                rot_base_T_cam_wxyz = self.quat_xyzw_to_wxyz(rot_base_T_cam_xyzw)

                # Get current EE pose in camera frame
                (trans_cam_T_ee, rot_cam_T_ee_xyzw) = self.listener.lookupTransform('/camera_link', '/panda_EE', rospy.Time(0))
                # Convert quaternion from xyzw to wxyz
                rot_cam_T_ee_wxyz = self.quat_xyzw_to_wxyz(rot_cam_T_ee_xyzw)

                # Joint angle information
                # Get the current joint state (so that it is not overwritten while recording)
                joint_state = deepcopy(self.joint_state)
                joint_position = joint_state.position
                joint_velocity = joint_state.velocity
                joint_effort = joint_state.effort

                # Tracked bbox information (in pixels)
                # Convert bbox to range [0.0-1.0]
                curr_bbox_xyxy4 = np.array([float(self.bbox.xmin/self.img_width), 
                                            float(self.bbox.ymin/self.img_height),
                                            float(self.bbox.xmax/self.img_width), 
                                            float(self.bbox.ymax/self.img_height)])

                demo_data_now = np.concatenate([
                    np.array(trans_base_T_ee),
                    np.array(rot_base_T_ee_wxyz),
                    np.array(trans_base_T_cam),
                    np.array(rot_base_T_cam_wxyz),
                    np.array(trans_cam_T_ee),
                    np.array(rot_cam_T_ee_wxyz),
                    np.array(curr_bbox_xyxy4),
                    np.array(joint_position),
                    np.array(joint_velocity),
                    np.array(joint_effort),
                    ], 
                    axis=0)
                
                assert demo_data_now.shape == (52,)

                self.demonstration[counter_] = demo_data_now

                # Save image
                # Convert your ROS Image message to OpenCV2
                try:
                    # Convert your ROS Image message to OpenCV2
                    cv2_img = self.cv_bridge.imgmsg_to_cv2(self.color_image, "bgr8")
                    
                    # Make sure that the image size is correct
                    rcvd_img_width = cv2_img.shape[1]
                    rcvd_img_height = cv2_img.shape[0]
                    assert rcvd_img_width==self.img_width and rcvd_img_height==self.img_height, f'Image size is wrong: {rcvd_img_width}x{rcvd_img_height}'
                except CvBridgeError as e:
                    print(e)
                else:
                    # Save your OpenCV2 image as a jpeg 
                    time_stamp = self.color_image.header.stamp
                    img_name = 'color_img_' + 'c' + '{0:04d}_'.format(counter_) + 't' + str(time_stamp) + '.jpeg'
                    img_path = os.path.join(self.image_save_dir, img_name)
                    res = cv2.imwrite(img_path, cv2_img)
                    print('Image saved: ' + str(res))

                print("Step", counter_)
                print("EE position in base frame=", list(trans_base_T_ee))
                print("EE quaternion in base frame=", list(rot_base_T_ee_wxyz))
                print("Bbox=", list(curr_bbox_xyxy4))
                print(self.color_image.header)
                print("#########")
                counter_ += 1

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue 

            self.rate.sleep()
        
        # Save recorded demonstration
        np.savetxt(self.robot_filepath, 
                   self.demonstration)
        
        demo_duration = time.time() - start
        print("Duration=", demo_duration)
        print("Demonstration saved in " + self.robot_filepath)

        # Save metadata
        # First store args
        metadata_dict = vars(args)
        # Store demo duration
        metadata_dict['duration'] = demo_duration
        self.write_json_file(self.metadata_filepath, 
                             metadata_dict)

if __name__ == '__main__':
    
    args = parse()
    recorder = FrankaRecorder(args)
    recorder.record()
    
