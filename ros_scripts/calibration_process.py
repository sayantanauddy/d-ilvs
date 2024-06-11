import os
import numpy as np
from cv2 import calibrateHandEye

def get_files_with_extension(directory, extension):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_list.append(os.path.join(root, file))
    return file_list

save_dir = 'calibration_data'

# Get a list of .npz files in save_dir
cal_files = get_files_with_extension(directory=save_dir, 
                                     extension='.npz')

R_target2cam_list = list()
t_target2cam_list = list()
R_gripper2base_list = list()
t_gripper2base_list = list()

for cal_file in cal_files:
    print(f'Processing {cal_file}')

    # Load data
    loaded_data = np.load(cal_file)
    R_target2cam_list.append(loaded_data['R_target2cam'])
    t_target2cam_list.append(loaded_data['t_target2cam'])
    R_gripper2base_list.append(loaded_data['R_gripper2base'])
    t_gripper2base_list.append(loaded_data['t_gripper2base'])

R_target2cam_list = np.array(R_target2cam_list)
t_target2cam_list = np.array(t_target2cam_list)
R_gripper2base_list = np.array(R_gripper2base_list)
t_gripper2base_list = np.array(t_gripper2base_list)

# Compute hand-eye
R_cam2gripper, t_cam2gripper = calibrateHandEye(
    R_target2cam=R_target2cam_list,
    t_target2cam=t_target2cam_list,
    R_gripper2base=R_gripper2base_list,
    t_gripper2base=t_gripper2base_list
)

print(f'Processed {len(cal_files)} observations')

print('R_cam2gripper=')
print(R_cam2gripper)

print('t_cam2gripper=')
print(t_cam2gripper)

# Save results
np.save(os.path.join(save_dir, 'R_cam2gripper.npy'), R_cam2gripper)
np.save(os.path.join(save_dir, 't_cam2gripper.npy'), t_cam2gripper)

