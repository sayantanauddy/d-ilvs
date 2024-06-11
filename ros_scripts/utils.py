import numpy as np
from os.path import dirname, join as pjoin


def quaternion_multiply(q1_wxyz, q2_wxyz):
    w0, x0, y0, z0 = q2_wxyz
    w1, x1, y1, z1 = q1_wxyz
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                      x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                      x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=q1_wxyz.dtype)

def computeLvPinv(e, Ls):
    
    normE = np.linalg.norm(e)
    
    LsLsT = Ls @ Ls.transpose()
    
    eTLsLsT = e.transpose() @ LsLsT
    den = eTLsLsT @ e
    
    eTLs = e.transpose() @ Ls
    
    Lv = eTLs / normE

    LsTe = Ls.transpose() @ e

    LvPinv = (normE * LsTe ) / den

    LsTeeTLs = LsTe @ eTLs

    size_I = Ls.shape[1]
    Pv = np.eye(size_I) - (LsTeeTLs/den)

    return Lv, LvPinv, Pv
    
def switchingRule(errNorm, e0, e1, l0=0, l1=1):

    gain = 1 / (1 + np.exp(-12*((errNorm - e0)/(e1 - e0))+6))

    gainBar = (gain - l0)/(l1 - l0)
    if errNorm < e0:
        gainBar = 0
    elif errNorm > e1:
        gainBar = 1
    
    return gainBar
    
def get_camera_parameters():
    
    # Intrinsic parameters
    intrinsic = {
        'fx': 609.1783,
        'fy': 611.4904,
        'cx': 315.01038,
        'cy': 246.18089,
        'image_height': 480,
        'image_width': 640
        }

    # Extrinsic parameters, camera pose w.r.t. end-effector
    ee_T_c = np.array([
            [0.01052516547, -0.9983949956, 0.05564758461, 0.03289467945],
            [0.9999441178, 0.01045369622, -0.00157525815, -0.03571680812],
            [0.0009910069091, 0.05566105475, 0.99844923, -0.03371846965],
            [0, 0, 0, 1]])

    return intrinsic, ee_T_c