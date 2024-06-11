#!/usr/bin/env python  
from collections import deque
import argparse
import rospy
from darknet_ros_msgs.msg import BoundingBox, BoundingBoxes

# How to run: 
# 1. On robot computer:
# roscore
# (optional) roslaunch franka_example_controllers cartesian_impedance_example_controller.launch robot_ip:=192.168.64.147 load_gripper:=true
# 
# On camera computer: 
# 2. roslaunch realsense2_camera rs_camera.launch color_width:=640 color_height:=480 color_fps:=30
# 3. roslaunch darknet_ros darknet_ros.launch
# 4. python ros_scripts/bbox_tracker.py --tracked_obj_class mouse --prob_thresh 0.4 --hist_len 50

# YOLO object classes are given in: https://github.com/leggedrobotics/darknet_ros

def parse():

    parser = argparse.ArgumentParser(description='Collect demos on Franka')

    parser.add_argument('--prob_thresh', type=float, default=0.7, help='Probability threshold for bbox detection')
    parser.add_argument('--tracked_obj_class', type=str, default='cup', help='Object to track')    
    parser.add_argument('--freq', type=int, default=40, help='Frequency in Hz')
    parser.add_argument('--hist_len', type=int, default=150, help='Number of frames of history for smoothing tracked bbox')
    parser.add_argument('--noclass', action='store_true', help='Ignore object class and track the object with highest confidence')    
    parser.set_defaults(noclass=False) 

    args = parser.parse_args()
    return args


class BboxTracker():

    def __init__(
            self,
            prob_thresh:float=0.7,
            tracked_obj_class:str='cup',
            freq:int=40,
            hist_len:int=150,  # Number of frames of history for smoothing tracked bbox
            ):

        # Ignore objects with lower confidence
        self.prob_thresh = prob_thresh

        self.tracked_obj_class = tracked_obj_class

        # To keep track of frames
        self.counter = 0

        # Flag to check if first object has been detected
        # Should be set to True only once and not changed again
        self.first_obj_detected = False

        # Current tracked bbox
        self.curr_bbox = None

        # History of bboxes for smoothing
        # The last element is the most recent
        self.bbox_hist_fifo = deque(maxlen=hist_len)
        self.smooth_bbox = None

        # Number of objects detected in the current frame
        self.curr_num_objects = 0

        # Initialize this node
        rospy.init_node("bbox_tracker")

        self.rate = rospy.Rate(freq)

        # Subscribe for getting bboxes
        rospy.Subscriber('/darknet_ros/bounding_boxes', 
                         BoundingBoxes, 
                         self.bbox_callback)

        # Publisher for publishing the tracked bbox
        self.pub = rospy.Publisher('bbox_tracker', 
                                   BoundingBox, 
                                   queue_size=1)

    def publish_tracked_bbox(self):
        while not rospy.is_shutdown():

            # Do not start publishing till first object is detected
            if not self.first_obj_detected:
                print('No object detected till now')
            else:    
                # Publish the tracked object
                self.pub.publish(self.curr_bbox)       

                print(f'Counter: {self.counter}', 
                      f'Number of objects: {self.curr_num_objects}', 
                      f'Object with highest prob: {self.curr_bbox.Class}',
                      f'xmin:{self.curr_bbox.xmin}, ymin:{self.curr_bbox.ymin}, xmax:{self.curr_bbox.xmax}, ymax:{self.curr_bbox.ymax}, prob:{self.curr_bbox.probability}',
                      f'Smooth bbox: {self.smooth_bbox}')

            # Increment the counter
            self.counter += 1

            self.rate.sleep()

    def get_smooth_bbox(self):

        # Smooth the history to find the published bbox
        xmin_smooth = 0
        ymin_smooth = 0
        xmax_smooth = 0
        ymax_smooth = 0
        for bbox in self.bbox_hist_fifo:
            xmin_smooth += bbox.xmin
            ymin_smooth += bbox.ymin
            xmax_smooth += bbox.xmax
            ymax_smooth += bbox.ymax
        xmin_smooth = round(float(xmin_smooth)/len(self.bbox_hist_fifo))
        ymin_smooth = round(float(ymin_smooth)/len(self.bbox_hist_fifo))
        xmax_smooth = round(float(xmax_smooth)/len(self.bbox_hist_fifo))
        ymax_smooth = round(float(ymax_smooth)/len(self.bbox_hist_fifo))

        smooth_bbox = BoundingBox(xmin=xmin_smooth,
                                  ymin=ymin_smooth,
                                  xmax=xmax_smooth,
                                  ymax=ymax_smooth)
        
        return smooth_bbox
    
    def set_equal_aspect(self, bbox:BoundingBox):
        # Converts bounding box to a square

        # Find the width and height of the bbox
        bbox_width = bbox.xmax - bbox.xmin
        bbox_height = bbox.ymax - bbox.ymin

        # Find the longer side
        bbox_longer_side = bbox_width if bbox_width > bbox_height else bbox_height

        # Find the centroid
        bbox_center_x = (bbox.xmin + bbox.xmax)//2
        bbox_center_y = (bbox.ymin + bbox.ymax)//2

        xmin_square = bbox_center_x - bbox_longer_side//2
        xmax_square = bbox_center_x + bbox_longer_side//2

        ymin_square = bbox_center_y - bbox_longer_side//2
        ymax_square = bbox_center_y + bbox_longer_side//2

        square_bbox = BoundingBox(xmin=xmin_square,
                                  ymin=ymin_square,
                                  xmax=xmax_square,
                                  ymax=ymax_square)
        
        return square_bbox

    def bbox_callback(
            self, 
            data:BoundingBoxes):
        
        # Detected bounding boxes sorted by proability (highest prob first)
        curr_bounding_boxes = sorted(
            data.bounding_boxes, 
            key=lambda bbox: bbox.probability,
            reverse=True)   

        if args.noclass:
            # The object with the highest probability
            curr_bbox = curr_bounding_boxes[0]
        else:
            # Select the desired object
            curr_bbox = BoundingBox(probability=0.0)
            for bbox in curr_bounding_boxes:
                if bbox.Class == self.tracked_obj_class:
                    curr_bbox = bbox
                    break

        if curr_bbox.probability >= self.prob_thresh:

            # Make the aspect ratio of bbox equal (square)
            curr_bbox = self.set_equal_aspect(curr_bbox)

            # Insert into the FIFO history
            self.bbox_hist_fifo.append(curr_bbox)

            # First object is detected with confidence
            self.first_obj_detected = True

            # Count number of objects in current frame        
            self.curr_num_objects = len(curr_bounding_boxes)

            # Set the bbox to be published
            self.smooth_bbox = self.get_smooth_bbox()
            self.curr_bbox = self.smooth_bbox

if __name__ == '__main__':

    args = parse()
    bbox_tracker = BboxTracker(
        prob_thresh=args.prob_thresh,
        tracked_obj_class=args.tracked_obj_class,
        freq=args.freq,
        hist_len=args.hist_len)
    bbox_tracker.publish_tracked_bbox()
