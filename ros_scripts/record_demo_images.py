#!/usr/bin/env python  
import roslib
import rospy
import math
import tf
import time
import datetime 
import numpy as np
import geometry_msgs.msg
import actionlib
import rospkg
import argparse
import os
from cv_bridge import CvBridge, CvBridgeError
import cv2
from copy import deepcopy
from sensor_msgs.msg import JointState, Image

def parse():

    parser = argparse.ArgumentParser(description='Collect demos on Franka')

    parser.add_argument('--freq', type=float, default=40.0, help='Frequency in Hz')
    parser.add_argument('--n_pts', type=int, default=500, help='How many data points to collect')
    parser.add_argument('--robot_filename', type=str, help='File name to save robot demos in')    
    parser.add_argument('--savedir_root', type=str, default='data_img_joint_ee', help='Root dir for saving')

    args = parser.parse_args()
    return args


class FrankaRecorder:

    def __init__(self, args):
        self.freq = args.freq
        self.n_pts = args.n_pts
        self.robot_filename = args.robot_filename

        self.joint_state = None
        self.color_image = None

        # Current timestamp
        now = datetime.datetime.now()
        timestamp_str = now.strftime('%Y%m%d_%H%M%S')

        dir_name = 'demo_' + str(timestamp_str)
        self.save_dir = os.path.join(args.savedir_root,
                                     dir_name)
        self.image_save_dir = os.path.join(self.save_dir, 'images')

        # Create save directories           
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.image_save_dir):
            os.makedirs(self.image_save_dir)

        # Position - 3
        # Orientation (quaternions) - 4
        # Joint positon - 9
        # Joint velocity - 9
        # Joint effort - 9
        self.demonstration = np.ndarray(shape=(args.n_pts, 34))

    def image_callback(self, data):
        self.color_image = data        

    def joint_states_callback(self, data):
        self.joint_state = data

    def record(self):

        rospy.init_node('save_pose_data')

        # For getting end-effector pose and orientation
        listener = tf.TransformListener()
        rate = rospy.Rate(args.freq)

        # For getting joint angles
        rospy.Subscriber('joint_states', JointState, self.joint_states_callback)

        # For getting images
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        # Instantiate bridge
        bridge = CvBridge()

        n_pts = args.n_pts

        counter_ = 0

        #start = time.time()
        while not rospy.is_shutdown() and counter_ < n_pts:
            try:	
                # Robot pose
                (trans,rot) = listener.lookupTransform('/panda_link0', '/panda_EE', rospy.Time(0))
                
                # Robot orientation
                self.demonstration[counter_, 0:3] = trans
                self.demonstration[counter_, 3] = rot[3]
                self.demonstration[counter_, 4:7] = rot[0:3]

                # Joint angle information
                # Get the current joint state (so that it is not overwritten while recording)
                joint_state = deepcopy(self.joint_state)
                self.demonstration[counter_, 7:16] = joint_state.position
                self.demonstration[counter_, 16:25] = joint_state.velocity
                self.demonstration[counter_, 25:34] = joint_state.effort

                # Save image
                # Convert your ROS Image message to OpenCV2
                try:
                    # Convert your ROS Image message to OpenCV2
                    cv2_img = bridge.imgmsg_to_cv2(self.color_image, "bgr8")
                    print(cv2_img.shape)
                except CvBridgeError as e:
                    print(e)
                else:
                    # Save your OpenCV2 image as a jpeg 
                    time = self.color_image.header.stamp
                    img_name = 'color_img_' + 'c' + '{0:04d}_'.format(counter_) + 't' + str(time) + '.jpeg'
                    img_path = os.path.join(self.image_save_dir, img_name)
                    res = cv2.imwrite(img_path, cv2_img)
                    print('Image saved: ' + str(res))

                print("Step", counter_)
                print("Position=", list(self.demonstration[counter_, 0:3]))
                print("Orientation=", list(self.demonstration[counter_, 3:7]))
                #print("Joint position=", list(self.demonstration[counter_, 7:16]))
                #print("Joint velocity=", list(self.demonstration[counter_, 16:25]))
                #print("Joint effort=", list(self.demonstration[counter_, 25:34]))
                print(self.color_image.header)
                print("#########")
                counter_ += 1

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                print(e)
                continue 

            rate.sleep()
        
        robot_filename = os.path.join(self.save_dir, args.robot_filename+'.txt')
        np.savetxt(robot_filename, self.demonstration)
        #print("Duration=", time.time() - start)
        print("Demonstration saved in " + robot_filename)

if __name__ == '__main__':
    
    args = parse()
    recorder = FrankaRecorder(args)
    recorder.record()
    
