from collections import namedtuple
import torch

from FrEIA.framework import *
from FrEIA.modules import *

import sys
sys.path.insert(0, '../../')
from hint import *

from data import PlusShapeModel as model
from data import prepare_data_loaders
n_parameters, n_observations = model.n_parameters, model.n_observations


######################################################
###   TRAINING CONFIGURATION AND HYPERPARAMETERS   ###
######################################################

c = {
    # GENERAL STUFF
    'suffix': f'{model.name}_conditional_hint-4-full', # identifier for trained models and outputs
    'device': 'cuda', # 'cuda' for GPU, 'cpu' for CPU
    'interactive_visualization': True, # requires visdom package to be installed

    # DATA
    'ndim_x': n_parameters,
    'ndim_y': n_observations,
    'ndim_z': n_parameters + n_observations,
    'data_model': model(),
    'vis_y_target': (0.83101039, 0.43478332, 2.32117294),

    # MODEL ARCHITECTURE
    'n_blocks': 4,
    # 'hidden_layer_sizes': 182, # 3M
    'hidden_layer_sizes': 212, # 4M
    'init_scale': 0.005,

    # TRAINING HYPERPARAMETERS
    'n_epochs': 50, # total number of epochs
    'max_batches_per_epoch': 100, # iterations per epoch (or if training data exhausted)
    'batch_size': 10000,
    'n_test': 100000,
    'n_train': 1000000,

    'lr_init': 0.01, # initial learning rate
    'pre_low_lr': 3, # number of epochs at the start with very low learning rate
    'final_decay': 0.01, # fraction of the learning rate to reach at final epoch
    'l2_weight_reg': 1.86e-05, # strength of the weight regularization
    'adam_betas': (0.9, 0.95),
}

# DATA LOADERS
train_loader, test_loader = prepare_data_loaders(c['data_model'], c['n_train'], c['n_test'], c['batch_size'])
c['train_loader'] = train_loader
c['test_loader'] = test_loader

# create namedtuple from config dictionary
c = namedtuple("Configuration",c.keys())(*c.values())
assert (c.ndim_x + c.ndim_y == c.ndim_z), "Dimensions don't match up!"


##############################
###   MODEL ARCHITECTURE   ###
##############################

y_lane = [InputNode(c.ndim_y, name='y')]
x_lane = [InputNode(c.ndim_x, name='x')]

for i in range(c.n_blocks):
    if i > 0:
        y_lane.append(Node(y_lane[-1],
                           HouseholderPerm,
                           {'fixed': False, 'n_reflections': c.ndim_y},
                           name=f'perm_y_{i}'))
        x_lane.append(Node(x_lane[-1],
                           HouseholderPerm,
                           {'fixed': False, 'n_reflections': c.ndim_x},
                           name=f'perm_x_{i}'))

    x_lane.append(Node(x_lane[-1],
                       HierarchicalAffineCouplingBlock,
                       {'c_internal': [c.hidden_layer_sizes, c.hidden_layer_sizes//2, c.hidden_layer_sizes//4]},
                       name=f'hac_x_{i+1}'))

    if i < c.n_blocks-1:
        x_lane.append(Node(x_lane[-1],
                           ExternalAffineCoupling,
                           {'F_class': F_fully_connected,
                            'F_args': {'internal_size': c.hidden_layer_sizes}},
                           conditions=y_lane[-1],
                           name=f'ac_y_to_x_{i+1}'))

    y_lane.append(Node(y_lane[-1],
                       AffineCoupling,
                       {'F_class': F_fully_connected,
                        'F_args': {'internal_size': c.hidden_layer_sizes}},
                       name=f'ac_y_{i+1}'))

y_lane.append(OutputNode(y_lane[-1], name='z_y'))
x_lane.append(OutputNode(x_lane[-1], name='z_x'))

model = ReversibleGraphNet(y_lane + x_lane, verbose=False)
model.to(c.device)


def model_inverse(test_y, test_z):
    z_y, z_x = model([test_y, torch.randn(test_y.shape[0], n_parameters).to(c.device)])
    y_test, x_test = model([z_y, test_z], rev=True)
    return x_test

def sample_joint(n_samples):
    return model([torch.randn(n_samples, n_observations).to(device),
                  torch.randn(n_samples, n_parameters).to(device)], rev=True)

def sample_conditional(y, z_x=None):
    if z_x is None:
        z_x = torch.randn(y.shape[0], n_parameters).to(device)
    z_y, _ = model([y, z_x])
    y, x = model([z_y, z_x], rev=True)
    return x
