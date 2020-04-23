from collections import namedtuple
import torch

from FrEIA.framework import *
from FrEIA.modules import *

import sys
sys.path.insert(0, '../../')
from hint import *

from data import prepare_uci_loaders, Gas as model
n_parameters = model.n_parameters


######################################################
###   TRAINING CONFIGURATION AND HYPERPARAMETERS   ###
######################################################

c = {
    # GENERAL STUFF
    'suffix': f'{model.name}_hint-8', # identifier for trained models and outputs
    'device': 'cuda', # 'cuda' for GPU, 'cpu' for CPU
    'interactive_visualization': True, # requires visdom package to be installed

    # DATA
    'ndim_x': n_parameters,
    'ndim_y': 0,
    'ndim_z': n_parameters,
    'data_model': model(),
    'vis_y_target': None,

    # MODEL ARCHITECTURE
    'n_blocks': 8,
    'hidden_layer_sizes': 128, # 500k
    'init_scale': 0.005,

    # TRAINING HYPERPARAMETERS
    'n_epochs': 50, # total number of epochs
    'max_batches_per_epoch': 1000, # iterations per epoch (or if training data exhausted)
    'batch_size': 853,

    'lr_init': 0.01, # initial learning rate
    'pre_low_lr': 3, # number of epochs at the start with very low learning rate
    'final_decay': 0.01, # fraction of the learning rate to reach at final epoch
    'l2_weight_reg': 1.86e-05, # strength of the weight regularization
    'adam_betas': (0.9, 0.95),
}

# DATA LOADERS
train_loader, test_loader = prepare_data_loaders(c['data_model'].name, c['batch_size'])
c['train_loader'] = train_loader
c['test_loader'] = test_loader

# create namedtuple from config dictionary
c = namedtuple("Configuration",c.keys())(*c.values())
assert (c.ndim_x + c.ndim_y == c.ndim_z), "Dimensions don't match up!"


##############################
###   MODEL ARCHITECTURE   ###
##############################

x_lane = [InputNode(c.ndim_x, name='x')]

for i in range(c.n_blocks):
    if i > 0:
        x_lane.append(Node(x_lane[-1],
                           HouseholderPerm,
                           {'fixed': True, 'n_reflections': c.ndim_x},
                           name=f'perm_{i}'))

    x_lane.append(Node(x_lane[-1],
                       HierarchicalAffineCouplingBlock,
                       {'c_internal': [c.hidden_layer_sizes, c.hidden_layer_sizes//2, c.hidden_layer_sizes//4, c.hidden_layer_sizes//8]},
                       name=f'hac_{i+1}'))

x_lane.append(OutputNode(x_lane[-1], name='z'))

model = ReversibleGraphNet(x_lane, verbose=False)
model.to(c.device)


def model_inverse(test_z):
    return model(test_z, rev=True)
