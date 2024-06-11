import logging
from typing import Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from tqdm import trange

class VisionImitationModel(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 hidden_dim: int, 
                 num_hidden_layers: int, 
                 dropout:float=0.5,
                 enforce_unit_quat=False,
                 verbose:bool=True) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout=dropout
        self.enforce_unit_quat = enforce_unit_quat
        self.verbose = verbose

        # Construct the layers
        self.layers = nn.Sequential()
        # Input layer
        self.layers.add_module('layer_in', nn.Linear (input_dim, self.hidden_dim))
        self.layers.add_module(f'drop_in', nn.Dropout(self.dropout))
        self.layers.add_module(f'relu_in', nn.ReLU())
        # Hidden layers
        for l in range(1, self.num_hidden_layers):
            self.layers.add_module(f'layer{l}', nn.Linear(self.hidden_dim, self.hidden_dim))
            self.layers.add_module(f'drop_{l}', nn.Dropout(self.dropout))
            self.layers.add_module(f'relu_{l}', nn.ReLU())
        # Output layer
        self.layers.add_module('layer_out', nn.Linear(self.hidden_dim, self.output_dim))
        self.layers.add_module('tanh_out', nn.Tanh())

        # Display network architecture
        if self.verbose:
            self.print_model_arch()

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:

        output = self.layers(x)

        # Enforce unit quaternion
        if self.enforce_unit_quat:
            output[:, 3:] = nn.functional.normalize(output[:, 3:], dim=1)

        return output

    def print_model_arch(self):

        total_train_params = 0
        total_notrain_params = 0

        layer_info = list()
        for n,p in self.named_parameters():
            layer_info.append(dict(layer_name=n,
                                   shape=list(p.shape),
                                   total_params=np.prod(p.shape),
                                   trainable=p.requires_grad,
                                   ))
            if p.requires_grad:
                total_train_params += np.prod(p.shape)
            else:
                total_notrain_params += np.prod(p.shape)

        # Dataframe of architecture info
        arch_df = pd.DataFrame(layer_info)

        logging.info('#### Layer details')
        logging.info('\n'+ str(arch_df))

        logging.info('#### Model summary')
        logging.info(f'Input dimension: {self.input_dim}')
        logging.info(f'Ouput dimension: {self.output_dim}')
        logging.info(f'Hidden dimension: {self.hidden_dim}')
        logging.info(f'Number of hidden layers: {self.num_hidden_layers}')
        logging.info(f'Trainable parameters: {total_train_params}')
        logging.info(f'Non-trainable parameters: {total_notrain_params}')
        logging.info(f'Total parameters: {total_train_params+total_notrain_params}')

def integrate(ode_rhs,x0,t,rtol=1e-6,atol=1e-7, method='dopri5'):
    ''' Performs forward integration with Dopri5 (RK45) solver with rtol=1e-6 & atol=1e-7.
        Higher-order solvers as well as smaller tolerances would give more accurate solutions.
        Inputs:
            ode_rhs    time differential function with signature ode_rhs(t,s)
            x0 - [N,d] initial values
            t  - [T]   integration time points
        Retuns:
            xt - [N,T,d] state trajectory computed at t
    '''
    return odeint(ode_rhs, x0, t, method=method, rtol=rtol, atol=atol).permute(1,0,2)

class NODE(nn.Module):
    def __init__(self, target_network, explicit_time=0, method='dopri5', verbose=False):
        ''' d - ODE dimensionality '''
        super().__init__()
        self.set_target_network(target_network)
        self.explicit_time = explicit_time
        self.method = method

        if verbose:
            import numpy as np
            total_params = 0
            print('NODE parameters:')
            for n,p in self.target_network.named_parameters():
                print(n,p.shape)
                total_params += np.prod(list(p.shape))
            print(f'Total parameters: {total_params}')

    def set_target_network(self, target_network):
        self.target_network = target_network
    
    @property
    def ode_rhs(self):
        ''' returns the differential function '''
        if self.explicit_time == 1:
            return lambda t, x: self.target_network(torch.cat([x, 
                                                               t.repeat(*(list(x.shape[0:-1])+[1]))
                                                              ], 
                                                            dim=-1))
        elif self.explicit_time == 0:
            return lambda t, x: self.target_network(torch.cat([x, 
                                                              ], 
                                                              dim=-1))
        else:
            raise NotImplementedError(f'Invalid value of explicit_time={self.explicit_time} (only 0 or 1 allowed)')
    
    def forward(self, t, x0):
        ''' Forward integrates the NODE system and returns state solutions
            Input
                t  - [T]   time points
                x0 - [N,d] initial value
            Returns
                X  - [N,T,d] forward simulated states
        '''
        return integrate(self.ode_rhs, x0, t, method=self.method).float()