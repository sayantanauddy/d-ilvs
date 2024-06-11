#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyRequest

def call_franka_gripper_open_service():
    rospy.wait_for_service('franka_gripper_open')
    
    try:
        franka_gripper_open = rospy.ServiceProxy('franka_gripper_open', Empty)
        response = franka_gripper_open()
        print("Franka gripper open service called successfully!")
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")

if __name__ == "__main__":
    rospy.init_node('franka_gripper_open_client')
    
    # Call the service
    call_franka_gripper_open_service()
