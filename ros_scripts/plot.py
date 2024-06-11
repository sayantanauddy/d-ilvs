import numpy as np
import matplotlib.pyplot as plt

# This is temporary - file names are hard coded

stamp = '20230915_173825'

pos = np.load(f'plots_exp/robot_YOLO_pos_{stamp}.npy')
quat = np.load(f'plots_exp/robot_YOLO_quat_{stamp}.npy')
error = np.load(f'plots_exp/robot_YOLO_error_{stamp}.npy')

fig, ax = plt.subplots(3,1,figsize=(10,15),facecolor='white',sharex=True)
ax[0].plot(pos[:,0],label='p_x',color='red')
ax[0].plot(pos[:,1],label='p_y',color='c')
ax[0].plot(pos[:,2],label='p_z',color='b')
ax[1].plot(quat[:,0],label='q_w',color='r')
ax[1].plot(quat[:,1],label='q_x',color='c')
ax[1].plot(quat[:,2],label='q_y',color='b')
ax[1].plot(quat[:,3],label='q_z',color='k')
ax[2].plot(error,label='error',color='r')
ax[0].legend()
ax[1].legend()

ax[0].set_ylabel('Positions (m)')
ax[1].set_ylabel('Quaternions')
ax[2].set_ylabel('Errors')

print(f'first error={error[0]}, final error={error[-1]}')

fig.tight_layout()
fig.savefig(f'plots_exp/robot_YOLO_{stamp}.png')

