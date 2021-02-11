from math import exp

import torch
import torch.nn as nn

from FrEIA.modules.orthogonal import HouseholderPerm



def linear_subnet_constructor(c_in, c_out, c_internal):
    return nn.Sequential(nn.Linear(c_in,       c_internal), nn.ReLU(),
                         nn.Linear(c_internal, c_internal), nn.ReLU(),
                         nn.Linear(c_internal, c_out))

def conv_subnet_constructor(c_in, c_out, c_internal):
    return nn.Sequential(nn.Conv2d(c_in,       c_internal, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(c_internal, c_internal, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(c_internal, c_out,      3, padding=1))


class HierarchicalAffineCouplingTree(nn.Module):
    '''TODO
    '''

    def __init__(self, data_shape, dims_c, conv=False, subnet_constructor=None, c_internal=[], clamp=2, max_splits=-1, min_split_size=2, reshuffle=False):
        super().__init__()
        self.data_shape = data_shape
        self.clamp = clamp
        if subnet_constructor is None:
            subnet_constructor = conv_subnet_constructor if conv else linear_subnet_constructor
        if len(c_internal) == 0:
            c_internal = [data_shape[0],]
        if len(c_internal) == 1:
            c_internal += c_internal

        if reshuffle:
            self.perm = HouseholderPerm([data_shape], dims_c=dims_c, n_reflections=data_shape[0], fixed=True)
        else:
            self.perm = None

        self.split_idx = data_shape[0] // 2
        self.conditional = (len(dims_c) > 0)
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])
        self.s = subnet_constructor(self.split_idx + condition_length, data_shape[0] - self.split_idx, c_internal[0])
        self.t = subnet_constructor(self.split_idx + condition_length, data_shape[0] - self.split_idx, c_internal[0])

        if data_shape[0] >= 2 * min_split_size and max_splits != 0:
            self.leaf = False
            self.upper = HierarchicalAffineCouplingTree((self.split_idx,) + data_shape[1:], dims_c,
                                                        conv, subnet_constructor, c_internal[1:], clamp, max_splits-1, min_split_size, reshuffle)
            self.lower = HierarchicalAffineCouplingTree((data_shape[0] - self.split_idx,) + data_shape[1:], dims_c,
                                                        conv, subnet_constructor, c_internal[1:], clamp, max_splits-1, min_split_size, reshuffle)
        else:
            self.leaf = True

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s))
    def log_e(self, s):
        '''log of the nonlinear function e'''
        return self.clamp * 0.636 * torch.atan(s)

    def forward(self, x, c=[], rev=False):
        # Potentially reshuffle
        if not rev and self.perm is not None:
            x = self.perm([x])[0]

        # Split data lanes
        x_upper, x_lower = torch.split(x, [self.split_idx, x.shape[1] - self.split_idx], dim=1)

        if (not self.leaf) and (not rev):
            # Recursively run subtree transformations
            x_upper, J_upper = self.upper.forward(x_upper, c=c, rev=rev)
            x_lower, J_lower = self.lower.forward(x_lower, c=c, rev=rev)

        # Compute own coupling transform and Jacobian
        x_upper_c = torch.cat([x_upper, *c], dim=1) if self.conditional else x_upper
        s, t = self.s(x_upper_c), self.t(x_upper_c)
        if not rev:
            x_lower = self.e(s) * x_lower + t
            J = self.log_e(s)
        else:
            x_lower = (x_lower - t) / self.e(s)
            J = -self.log_e(s)

        if (not self.leaf) and rev:
            # Reverse order of hierarchy during inverse pass
            x_upper, J_upper = self.upper.forward(x_upper, c=c, rev=rev)
            x_lower, J_lower = self.lower.forward(x_lower, c=c, rev=rev)

        x = torch.cat([x_upper, x_lower], dim=1)

        # Potentially reverse reshuffling
        if rev and self.perm is not None:
            x = self.perm([x], rev=True)[0]

        # Calculate block log Jacobian determinant
        J = torch.sum(J, dim=tuple(range(1, len(J.shape))))
        if not self.leaf:
            J = J_upper + J + J_lower

        return x, J


class HierarchicalAffineCouplingBlock(nn.Module):
    '''TODO
    '''

    def __init__(self, dims_in, dims_c=[], conv=False, subnet_constructor=None, c_internal=[], clamp=4., max_splits=-1, min_split_size=2, reshuffle=False):
        super().__init__()

        assert all([dims_c[i][1:] == dims_in[0][1:] for i in range(len(dims_c))]), \
            "Dimensions of input and one or more conditions don't agree."

        self.tree = HierarchicalAffineCouplingTree(dims_in[0],
                                                   dims_c=dims_c,
                                                   conv=conv,
                                                   subnet_constructor=subnet_constructor,
                                                   c_internal=c_internal,
                                                   clamp=clamp,
                                                   max_splits=max_splits,
                                                   min_split_size=min_split_size,
                                                   reshuffle=reshuffle)

    def forward(self, x, c=[], rev=False):
        x, self.jac = self.tree.forward(x[0], c, rev=rev)
        return [x]

    def jacobian(self, x, c=[], rev=False):
        return self.jac

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use one input."
        return input_dims
