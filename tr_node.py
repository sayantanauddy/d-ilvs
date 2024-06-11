import os
import numpy as np
import logging
import argparse
from tqdm import trange
from copy import deepcopy
from typing import Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from panda_vision_imitation.utils import set_seed, check_cuda
from panda_vision_imitation.logging import write_dict, write_np_dict, custom_logging_setup
from panda_vision_imitation.data import DataBBoxPositionQuaternion, get_minibatch_extended
from panda_vision_imitation.metrics import dtw_distance_fast
from panda_vision_imitation.model import NODE

def parse_args() -> argparse.Namespace:
    """_summary_

    Returns:
        argparse.Namespace: _description_
    """

    parser = argparse.ArgumentParser('Training a NODE for visual imitation')

    # Logging-related args
    parser.add_argument('--log_dir', type=str, required=True, help='Root folder for logs')
    parser.add_argument('--description', type=str, required=True, help='Description of experiment')

    # Data-related args    
    parser.add_argument('--data_root', type=str, required=True, help='Root dir of all dataset folders')
    parser.add_argument('--train_file', type=str, required=True, help='Training data file (.npy)')
    parser.add_argument('--val_file', type=str, required=True, help='Validation data file (.npy)')
    parser.add_argument('--tsub', type=int, default=30, help='Segment length for training')
    parser.add_argument('--start_deadzone', type=int, default=0, help='Deadzone to remove at the start of the trajectory')

    # Training-related args    
    parser.add_argument('--seed', type=int, required=True, help='Random seed for reproducability')
    parser.add_argument('--num_iter', type=int, required=True, help='Number of training iterations')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')

    # Model-related args    
    parser.add_argument('--hidden_dim', type=int, default=256, help='Number of hidden layer units')
    parser.add_argument('--num_hidden', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--explicit_time', type=int, default=0, help='Explicit time input for NODE')
    parser.add_argument('--rot_scale', type=float, required=True, help='Scaling factor for rotation vectors')

    args = parser.parse_args()
    return args

def main():

    # Get arguments
    args = parse_args()

    # Create logging folder and set up console logging
    save_dir, identifier = custom_logging_setup(args)

    # Check if GPU is available
    cuda_available, device = check_cuda()
    logging.info(f'cuda_available: {cuda_available}')

    # Set seed
    logging.info(f'seed: {args.seed}')
    set_seed(seed=args.seed)

    # Create training data object
    tr_data = DataBBoxPositionQuaternion(
        data_dir=args.data_root,
        datafile=args.train_file,
        norm=False,
        start_deadzone=args.start_deadzone,
        scale=args.rot_scale,
        device=device)
    
    # Create the target network (dynamical model inside the NODE)
    data_shape = tr_data.input.shape
    input_dim = data_shape[-1]
    output_dim = input_dim

    layers = OrderedDict()
    for layer_id in range(args.num_hidden):
        if layer_id == 0:
            in_features = input_dim + args.explicit_time
            out_features = args.hidden_dim
        elif layer_id == args.num_hidden - 1:
            in_features = args.hidden_dim
            out_features = output_dim
        else:
            in_features = args.hidden_dim
            out_features = args.hidden_dim 

        layers[f'layer{layer_id}'] = nn.Linear(in_features=in_features, out_features=out_features)
        if layer_id != args.num_hidden - 1:
            layers[f'relu{layer_id}'] = nn.ReLU()

    target_network = nn.Sequential(layers)

    # Initialize NODE
    node = NODE(target_network, 
                explicit_time=args.explicit_time, 
                method='euler', 
                verbose=True).to(device)
    
    # Initialize the optimizer
    theta_optimizer = optim.Adam(node.target_network.parameters(), 
                                 lr=args.lr)
    
    # Start training
    node.train()
    losses = list()
    for training_iters in trange(args.num_iter):
        theta_optimizer.zero_grad()
        t, y_all = get_minibatch_extended(tr_data.t, 
                                          tr_data.input, 
                                          nsub=None, 
                                          tsub=args.tsub, 
                                          dtype=torch.float)

        t = t.to(device)
        y_all = y_all.to(device)

        # Starting points
        y_start = y_all[:,0].float()

        # Predicted trajectories - forward simulation
        y_hat = node(t.float(), y_start)

        # MSE
        loss = ((y_hat-y_all)**2).mean()
        losses.append(loss.item())

        loss.backward()
        theta_optimizer.step()

    # Evaluate on training data
    node.eval()
    t = tr_data.t.float().to(device)
    y_start = tr_data.input[:,0].float().to(device)
    y_gt_tr = tr_data.input.float().to(device)

    # forward simulation
    print(t.shape, y_start.shape)
    y_hat_tr = node(t, y_start)

    dtw_train = dtw_distance_fast(
        y_gt_tr[:,:,0:3].detach().cpu().numpy(), 
        y_hat_tr[:,:,0:3].detach().cpu().numpy())

    # Evaluate on validation data
    val_data = DataBBoxPositionQuaternion(
        data_dir=args.data_root,
        datafile=args.val_file,
        norm=False,
        start_deadzone=args.start_deadzone,
        scale=args.rot_scale,
        device=device)
    
    t = val_data.t.float().to(device)
    y_start = val_data.input[:,0].float().to(device)
    y_gt_val = val_data.input.float().to(device)

    # forward simulation
    y_hat_val = node(t, y_start)

    dtw_val = dtw_distance_fast(
        y_gt_val[:,:,0:3].detach().cpu().numpy(), 
        y_hat_val[:,:,0:3].detach().cpu().numpy())
    
    logging.info(f'Evaluation resuts: dtw_train={dtw_train[0]}, dtw_val={dtw_val[0]}')

    # Save results
    results = dict(
        losses=list(losses),
        dtw_train=dtw_train[1].tolist(),
        dtw_val=dtw_val[1].tolist()
    )
    write_dict(
        os.path.join(save_dir, 'results.json'),
        results
    )

    np.savez(
        os.path.join(save_dir, 'predictions.npz'), 
        y_gt_tr=y_gt_tr.detach().cpu().numpy(),
        y_hat_tr=y_hat_tr.detach().cpu().numpy(),
        y_gt_val=y_gt_val.detach().cpu().numpy(),
        y_hat_val=y_hat_val.detach().cpu().numpy(),
        )

    # Save model
    torch.save(obj=node,
               f=os.path.join(save_dir, 'models', 'node.pt'))

if __name__ == '__main__':
    main()
