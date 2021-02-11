from collections import namedtuple
import torch

from FrEIA.framework import *
from FrEIA.modules import *

from data import PlusShapeModel as model
from data import prepare_data_loaders
n_parameters, n_observations = model.n_parameters, model.n_observations


######################################################
###   TRAINING CONFIGURATION AND HYPERPARAMETERS   ###
######################################################

c = {
    # GENERAL STUFF
    'suffix': f'{model.name}_conditional_recursive_cinn-4', # identifier for trained models and outputs
    'device': 'cuda', # 'cuda' for GPU, 'cpu' for CPU
    'interactive_visualization': True, # requires visdom package to be installed

    # DATA
    'ndim_x': n_parameters,
    'ndim_y': n_observations,
    'ndim_z': n_parameters,
    'data_model': model(),
    'vis_y_target': (0.75, 0.0, 1.0, 3.0),

    # MODEL ARCHITECTURE
    'n_blocks': 4,
    'hidden_layer_sizes': 267, # 4M
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


##############################
###   MODEL ARCHITECTURE   ###
##############################

nodes = [ConditionNode(c['ndim_y'], name='y')]
nodes.append(InputNode(c['ndim_x'], name='x'))

for i in range(c['n_blocks']):
    nodes.append(Node(nodes[-1],
                      HouseholderPerm,
                      {'fixed': True, 'n_reflections': c['ndim_x']},
                      name=f'perm_{i+1}'))
    nodes.append(Node(nodes[-1],
                      HierarchicalAffineCouplingBlock,
                      {'c_internal': [c['hidden_layer_sizes'], c['hidden_layer_sizes']//2, c['hidden_layer_sizes']//4]},
                      conditions=nodes[0],
                      name=f'ac_{i+1}'))

nodes.append(OutputNode(nodes[-1], name='z'))

model = ReversibleGraphNet(nodes, verbose=False)
model.to(c['device'])
model.params_trainable = list(filter(lambda p: p.requires_grad, model.parameters()))


def model_inverse(test_y, test_z):
    x_test = model([test_z], c=[test_y], rev=True)
    return x_test


c['model'] = model
c['model_inverse'] = model_inverse

# create namedtuple from config dictionary
c = namedtuple("Configuration",c.keys())(*c.values())
