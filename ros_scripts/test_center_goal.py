import numpy as np

# This is used by 
# robot_YOLO.py
# bbox_image_pub.py

def get_center_goal(image_width:int=640,
                    image_height:int=480,
                    pixel=False):
    
    # Hardcoded goal at center of image (square goal)
    test_xmin = 236
    test_ymin = 204
    test_xmax = 370
    test_ymax = 338

    test_goal_bbox_xyxy4 = None
    
    if not pixel:
        # Convert to range 0.0-1.0
        test_xmin = float(216)/image_width
        test_ymin = float(202)/image_height
        test_xmax = float(390)/image_width
        test_ymax = float(340)/image_height
        test_goal_bbox_xyxy4 = np.array([test_xmin, test_ymin, test_xmax, test_ymax])
    else:
        # Use original pixel values
        test_goal_bbox_xyxy4 = np.array([test_xmin, test_ymin, test_xmax, test_ymax])

    assert test_goal_bbox_xyxy4 is not None

    return test_goal_bbox_xyxy4
