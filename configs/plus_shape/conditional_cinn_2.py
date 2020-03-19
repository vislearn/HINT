from collections import namedtuple
import torch

from FrEIA.framework import *
from FrEIA.modules import *

import sys
sys.path.insert(0, '../../')
from hint import *

from data import FourierCurveModel as model
from data import prepare_data_loaders
n_parameters, n_observations = model.n_parameters, model.n_observations


######################################################
###   TRAINING CONFIGURATION AND HYPERPARAMETERS   ###
######################################################

c = {
    # GENERAL STUFF
    'suffix': f'{model.name}_conditional_cinn-2', # identifier for trained models and outputs
    'device': 'cuda', # 'cuda' for GPU, 'cpu' for CPU
    'interactive_visualization': True, # requires visdom package to be installed

    # DATA
    'ndim_x': n_parameters,
    'ndim_y': n_observations,
    'ndim_z': n_parameters,
    'data_model': model(),
    'vis_y_target': (0.52466008, 0.21816375, 2.29708147),

    # MODEL ARCHITECTURE
    'n_blocks': 2,
    'hidden_layer_sizes': 151, # 200k
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


##############################
###   MODEL ARCHITECTURE   ###
##############################

nodes = [ConditionNode(c.ndim_y, name='y')]
nodes.append(InputNode(c.ndim_x, name='x'))

for i in range(c.n_blocks):
    nodes.append(Node(nodes[-1],
                      HouseholderPerm,
                      {'fixed': False, 'n_reflections': c.ndim_x},
                      name=f'perm_{i+1}'))
    nodes.append(Node(nodes[-1],
                      AffineCoupling,
                      {'F_class': F_fully_connected,
                       'F_args': {'internal_size': c.hidden_layer_sizes}},
                      conditions=nodes[0],
                      name=f'ac_{i+1}'))

nodes.append(OutputNode(nodes[-1], name='z'))

model = ReversibleGraphNet(nodes, verbose=False)
model.to(c.device)


def model_inverse(test_y, test_z):
    x_test = model([test_z], c=[test_y], rev=True)
    return x_test
