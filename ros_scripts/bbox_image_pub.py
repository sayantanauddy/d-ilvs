#!/usr/bin/env python  
import argparse
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from darknet_ros_msgs.msg import BoundingBox 

# Publishes an image with the current YOLO bbox and goal bbox
# points marked in order

# How to run: 
# On camera computer: 
# 1. roslaunch realsense2_camera rs_camera.launch color_width:=640 color_height:=480 color_fps:=30
# 2. roslaunch darknet_ros darknet_ros.launch
# 3. python ros_scripts/bbox_tracker.py
# 4. python ros_scripts/bbox_image_pub.py --test_center_goal
#
# View published image:
# rosrun image_view image_view image:=/img_bbox_tracker

from test_center_goal import get_center_goal


def parse():

    parser = argparse.ArgumentParser(description='Publishes image of goal and current bbox')
    parser.add_argument('--test_center_goal', action='store_true', help='Test with the goal at center')    
    parser.set_defaults(test_center_goal=False) 

    args = parser.parse_args()
    return args

class BboxTracker():

    def __init__(
            self,
            goal_bbox,
            rate:int=40,
            ):

        self.rate = rospy.Rate(rate)

        # To keep track of frames
        self.counter = 0

        # Current tracked bbox
        self.curr_bbox = None

        # Current image
        self.curr_img = None

        self.goal_bbox = goal_bbox

        self.img = None

        self.br = CvBridge()
    
        # Subscribe for getting bboxes
        rospy.Subscriber('/bbox_tracker', 
                         BoundingBox, 
                         self.bbox_callback)

        # Subscribe to images
        rospy.Subscriber('/camera/color/image_raw', 
                         Image, 
                         self.img_callback)


        # Publisher for publishing the image with tracked bbox
        self.pub = rospy.Publisher('img_bbox_tracker', 
                                   Image, 
                                   queue_size=1)

    def bbox_callback(
            self, 
            data:BoundingBox):
        self.curr_bbox = data

    def img_callback(
            self,
            data:Image):
        self.curr_img = data

        # Convert to cv2
        img = self.br.imgmsg_to_cv2(self.curr_img)

        # Draw curr bbox
        cv2.putText(img, 'C1', (self.curr_bbox.xmin, self.curr_bbox.ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.putText(img, 'C2', (self.curr_bbox.xmin, self.curr_bbox.ymax), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.putText(img, 'C3', (self.curr_bbox.xmax, self.curr_bbox.ymax), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.putText(img, 'C4', (self.curr_bbox.xmax, self.curr_bbox.ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # Draw goal bbox
        cv2.putText(img, 'G1', (self.goal_bbox['xmin'], self.goal_bbox['ymin']), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        cv2.putText(img, 'G2', (self.goal_bbox['xmin'], self.goal_bbox['ymax']), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        cv2.putText(img, 'G3', (self.goal_bbox['xmax'], self.goal_bbox['ymax']), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        cv2.putText(img, 'G4', (self.goal_bbox['xmax'], self.goal_bbox['ymin']), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        self.img = self.br.cv2_to_imgmsg(img, encoding='rgb8')

    def main(self):

        while not rospy.is_shutdown():
            if self.img is not None:
                self.pub.publish(self.img)
            self.rate.sleep()

if __name__ == '__main__':
    # Initialize this node
    rospy.init_node("img_bbox_tracker_node")

    args = parse()

    if args.test_center_goal:
        # Goal at center of image
        test_bbox_center = get_center_goal(pixel=True)
        goal_bbox = dict(xmin=test_bbox_center[0], 
                        ymin=test_bbox_center[1], 
                        xmax=test_bbox_center[2], 
                        ymax=test_bbox_center[3])
    else:
        # Goal from training data (cup 2 last frame: raw_data/raw_cup_data_1/demo_20240117_161830/yolo_results.json)
        goal_bbox = {"xmin": 337, "ymin": 188, "xmax": 606, "ymax": 464}

    bbox_tracker = BboxTracker(goal_bbox)
    bbox_tracker.main()
