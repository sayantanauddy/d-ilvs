import os
import numpy as np
from sklearn import preprocessing
from pyquaternion import Quaternion

import torch
from torch.utils.data import Dataset

class VisionImitationDataset(Dataset):

    def __init__(self, 
                 data_dirs: list, 
                 data_file_name: str='network_data.txt',
                 vel_cmps=False,
                 x_mean=None,
                 x_scale=None,
                 ) -> None:
        """Dataset class for the Vision Imitation dataset

        Args:
            data_dirs (list): _description_
            data_file_name (str, optional): _description_. Defaults to 'network_train_data.txt'.
            x_mean (_type_, optional): _description_. Defaults to None.
            x_scale (_type_, optional): _description_. Defaults to None.
        """

        self.data_dirs = data_dirs
        self.data_file_name = data_file_name
        self.vel_cmps = vel_cmps

        data_list = list()
        for data_dir in self.data_dirs:
            # Read each data file
            data_path = os.path.join(data_dir, self.data_file_name)
            data_np = np.loadtxt(data_path)
            data_list.append(data_np)

        # Create the combined dataset
        self.data_np = np.concatenate(data_list)

        self.x = self.data_np[:,0:8]
        self.y = self.data_np[:,8:]

        # Normalize the input data
        if x_mean is None and x_scale is None:
            self.x_scaled, self.x_mean, self.x_scale= self.normalize(self.x)

            _x_scaled = (self.x - self.x_mean)/self.x_scale
            assert np.allclose(self.x_scaled, _x_scaled)
        else:
            self.x_mean = x_mean
            self.x_scale = x_scale
            self.x_scaled = (self.x - x_mean)/x_scale

    def normalize(self, x):
        scaler = preprocessing.StandardScaler().fit(x)
        x_mean = scaler.mean_
        x_scale = scaler.scale_
        x_scaled = scaler.transform(x)
        return x_scaled, x_mean, x_scale

    def __len__(self):
        return self.data_np.shape[0]

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'input': self.x_scaled[idx], 
                  'output': self.y[idx]}

        return sample
    

class DataBBoxPositionQuaternion():

    def __init__(self, 
                 data_dir, 
                 datafile, 
                 norm=True,
                 start_deadzone=0,
                 bbox_idx={'xmin': 0, 'ymin': 1, 'xmax': 2, 'ymax': 3},
                 position_idx={'x': 4, 'y': 5, 'z': 6}, 
                 rotation_idx={'w': 7, 'x': 8, 'y': 9, 'z': 10}, 
                 record_freq=40, 
                 device=torch.device('cpu'), 
                 scale=1.0):
        """
        All the data is first moved to the GPU (if available).
        Moving the data to the GPU may be more efficient.
        """
        self.data_dir = data_dir
        self.datafile = datafile
        self.device = device
        self.scale = scale
        self.record_freq = record_freq

        self.bbox_idx = bbox_idx
        self.norm = norm
        self.start_deadzone = start_deadzone

        # Position specific arguments
        self.pos_idx = position_idx

        # Rotation specific arguments
        self.rot_idx = rotation_idx
        
        # The trajectores and time stamps
        # Load the .npy file
        self.data = np.load(os.path.join(self.data_dir, self.datafile), allow_pickle=True)

        # Remove deadzones at the start of the trajectory
        self.data = self.data[:, self.start_deadzone:, :]

        bbox_idx_list = [self.bbox_idx[i] for i in ['xmin','ymin','xmax','ymax']]
        pos_idx_list = [self.pos_idx[i] for i in ['x','y','z']]
        rot_idx_list = [self.rot_idx[i] for i in ['w','x','y','z']]

        # The bboxes
        self.bboxes = self.data[:,:,bbox_idx_list]
        # The positions in 3D space
        self.position = self.data[:,:,pos_idx_list]
        # The 4D quaternions
        self.rotation_quat = self.data[:,:,rot_idx_list]

        # Find the number of sequences N and the length of each sequence T
        [self.num_demos, self.num_timesteps, self.quat_dim] = self.rotation_quat.shape
        [_, _, self.position_dim] = self.position.shape
        [_, _, self.bbox_dim] = self.bboxes.shape

        # Must load only quaternions
        assert self.position_dim == 3
        assert self.quat_dim == 4
        assert self.bbox_dim == 4

        #self.t = [np.linspace(0.0, self.num_timesteps, self.num_timesteps)/record_freq]
        self.t = np.linspace(0.0, 1.0, self.num_timesteps)

        # Project the quaternion trajectories to the tangent plane
        self.rotation = torch.from_numpy(self.to_tangent_plane())

        # Scale the values if needed
        self.rotation *= self.scale
        self.bboxes *= self.scale

        self.bboxes = torch.from_numpy(self.bboxes).to(device)

        # Normalize the position trajectories if needed
        # Convert to tensors and move to the required device
        if self.norm:
            self.position, self.position_mean, self.position_std = self.normalize(self.position)
            self.position, self.position_mean, self.position_std = torch.from_numpy(self.position), torch.from_numpy(self.position_mean), torch.from_numpy(self.position_std)
            self.position, self.position_mean, self.position_std = self.position.to(device), self.position_mean.to(device), self.position_std.to(device)
        else:
            self.position_mean, self.position_std = None, None
            self.position = torch.from_numpy(self.position).to(device)

        # Translating the position goal to the origin
        self.position_goal = self.position[:,-1,:]

        self.rotation = self.rotation.to(device)

        # input contains a concatenation of position and orientation (mapped to the tangent plane)
        self.input = torch.cat((self.bboxes, self.position, self.rotation), -1).to(device)
        self.t = torch.from_numpy(np.array(self.t)).to(device)

    def normalize(self, arr):
        """
        Normalizes the input array (only for positions)
        """
        # Compute the mean and std for x,y across all demonstration trajectories
        mean = np.expand_dims(np.mean(np.reshape(arr, (self.num_demos*self.num_timesteps, self.position_dim)), axis=0), axis=0)
        std = np.expand_dims(np.std(np.reshape(arr, (self.num_demos*self.num_timesteps, self.position_dim)), axis=0), axis=0)
        arr = (arr - mean)/std
        return arr, mean, std

    def denormalize(self, arr, type='pos'):
        """
        Denormalizes the input array
        """
        if not self.norm:
            return arr
            
        if type == 'pos':
            if not torch.is_tensor(arr):
                arr_ = torch.clone(torch.from_numpy(arr))
            else:
                arr_ = torch.clone(arr)
            arr_[:,:,0:3] = arr_[:,:,0:3]*self.position_std.cpu().detach().numpy() + self.position_mean.cpu().detach().numpy()
        else:
            raise NotImplementedError(f'Not implemented for {type}')

        # Return the same type as the argument (np array or torch tensor)
        if not torch.is_tensor(arr):
            # To make sure that only positions are denormalized
            assert torch.equal(torch.from_numpy(arr[:,:,3:]), arr_[:,:,3:])
            return arr_.detach().cpu().numpy()
        else:
            # To make sure that only positions are denormalized
            assert torch.equal(arr[:,:,3:], arr_[:,:,3:])
            return arr_

    def unnormalize(self, arr, type='pos'):
        return self.denormalize(arr, type)

    def to_tangent_plane(self):
        """
        Projects the demonstration quaternion trajectories to the Eucliden tangent plane
        """

        # The goal orientation
        q_goal = self.rotation_quat[:,-1,:]
        assert q_goal.shape == (self.num_demos, self.quat_dim)

        # Project quat_data to tangent plane using log map
        quat_data_projection = list()

        for demo in range(self.num_demos):
            # For each demo
            r_list = list()
            for step in range(self.num_timesteps):
                # Project a quaternion in each step
                q = Quaternion(q_goal[demo])
                p = Quaternion(self.rotation_quat[demo,step])

                r = Quaternion.log_map(q=q, p=p).elements

                # First value of r must be 0.0
                assert np.allclose(r[0], 0.0)

                # Remove the first element
                r = r[1:]
                r_list.append(r)
            quat_data_projection.append(r_list)

        quat_data_projection = np.array(quat_data_projection)

        # Check the shape
        assert quat_data_projection.shape == (self.num_demos, self.num_timesteps, 3)

        # The final point in the projected trajectory should be (0.0,0.0,0.0)
        # for each demonstration trajectory
        assert np.allclose(quat_data_projection[:,-1], np.zeros((self.num_demos, 3)))

        return quat_data_projection

    def from_tangent_plane(self, tangent_vector_data):
        """
        Projects the Eucliden tangent plane trajectories back to quaternion trajectories 
        """
        # The goal orientation
        q_goal = self.rotation_quat[:,-1,:]
        assert q_goal.shape == (self.num_demos, self.quat_dim)

        # Downscale if needed
        tangent_vector_data /= self.scale
        
        quat_data = list()
        for demo in range(self.num_demos):
            # For each demo
            q_list = list()
            for step in range(self.num_timesteps):
                q = Quaternion(q_goal[demo])
                r = tangent_vector_data[demo, step]
                assert r.shape == (3,)

                # r is 3-dimensional, insert 0.0 as the first element
                r = np.r_[np.zeros((1,)), r]
                assert r.shape == (4,)
                assert r[0] == 0.0
                r = Quaternion(r)
                
                try:
                    # Tangent vector projected back to quaternion
                    q_ = Quaternion.exp_map(q=q, eta=r)
                    # Enforce unit quaternions
                    if q_.norm > 0.0:
                        q_ /= q_.norm
                    q_ = q_.elements
                except Exception as e:
                    print(f'Exception: {e}')
                    print(f'q={q}')
                    print(f'r={r}')
                    print(f'q_={q_}')
                    q_ = q_list[-1]
                q_list.append(q_)

            quat_data.append(q_list)

        quat_data = np.array(quat_data)
        assert quat_data.shape == (self.num_demos, self.num_timesteps, 4)

        return quat_data

    def zero_center(self):
        """
        Center the position goal at the origin.
        Orientations remain unchanged.
        """
        # Create a copy of the data
        self.pos_goal_origin = torch.clone(self.pos)

        # Find the goal position
        self.position_goal = self.pos[:,-1,0:3]

        # Translate the goal position to the origin (only positions, not orientations)
        # self.pos_goal_origin should be used for training
        self.pos_goal_origin[:,:,0:3] -= torch.stack([self.position_goal]*self.pos.shape[1], axis=1)


    def unzero_center(self, arr):
        """
        Uncenter the position. Orientations remain unchanged.
        """
        # Translate the prediction away from the origin for (positions only)
        arr_ = torch.clone(arr)
        arr_[:,:,0:3] += torch.stack([self.position_goal]*self.pos.shape[1], axis=1)
        return arr_

def get_minibatch_extended(t, y, nsub=None, tsub=None, dtype=torch.float64):
    """
    Extract nsub sequences each of lenth tsub from the original dataset y.

    Args:
        t (np array [T]): T integration time points from the original dataset.
        y (np array [N,T,d]): N observed sequences from the original dataset, 
                              each with T datapoints where d is the dimension 
                              of each datapoint.
        nsub (int): Number of sequences to be selected from.
                    If Nsub is None, then all N sequences are considered.
        tsub (int): Length of sequences to be returned.
                    If tsub is None, then sequences of length T are returned.

    Returns:
        tsub (torch tensor [tsub]): Integration time points (in this minibatch)
        ysub (torch tensor [nsub, tsub, d]): Observed (sub)sequences (in this minibatch)
    """
    # Find the number of sequences N and the length of each sequence T
    [N,T] = y.shape[:2]

    # If nsub is None, then consider all sequences
    # Else select nsub sequences randomly
    y_   = y if nsub is None else y[torch.randperm(N)[:nsub]]

    # Choose the starting point of the sequences
    # If tsub is None, then start from the beginning
    # Else find a random starting point based on tsub
    t0   = 0 if tsub is None else torch.randint(0,1+len(t)-tsub,[1]).item()  # pick the initial value
    tsub = T if tsub is None else tsub

    # Set the data to be returned
    tsub, ysub = t[t0:t0+tsub], y_[:,t0:t0+tsub]

    if not torch.is_tensor(tsub):
        tsub = torch.from_numpy(tsub)

    if not torch.is_tensor(ysub):
        ysub = torch.from_numpy(ysub)

    return tsub, ysub 

# For transforming single quaternion to rotation vector
def to_tangent_plane(q_goal_np, q_np):
    """
    Projects a single quaternion to the Eucliden tangent plane
    """

    # Project a quaternion in each step
    q = Quaternion(q_goal_np)
    p = Quaternion(q_np)

    r = Quaternion.log_map(q=q, p=p).elements

    # First value of r must be 0.0
    assert np.allclose(r[0], 0.0)

    # Remove the first element
    r = r[1:]

    return r


# For transforming single rotation vector to a quaternion 
def from_tangent_plane(q_goal, r, scale=None):
    """
    Projects the Eucliden tangent plane trajectories back to quaternion trajectories 
    """

    # Downscale if needed
    if scale is not None:
        r /= scale
    
    q = Quaternion(q_goal)
    assert r.shape == (3,)

    # r is 3-dimensional, insert 0.0 as the first element
    r = np.r_[np.zeros((1,)), r]
    assert r.shape == (4,)
    assert r[0] == 0.0
    r = Quaternion(r)
    
    try:
        # Tangent vector projected back to quaternion
        q_ = Quaternion.exp_map(q=q, eta=r)
        # Enforce unit quaternions
        if q_.norm > 0.0:
            q_ /= q_.norm
        q_ = q_.elements
    except Exception as e:
        print(f'Exception: {e}')
        print(f'q={q}')
        print(f'r={r}')
        print(f'q_={q_}')

    return q_
