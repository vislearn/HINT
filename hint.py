
import torch
import torch.nn as nn



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

    def __init__(self, data_shape, conv=False, subnet_constructor=None, c_internal=[], clamp=2, max_splits=-1, min_split_size=2):
        super().__init__()
        self.data_shape = data_shape
        self.clamp = clamp
        if subnet_constructor is None:
            subnet_constructor = conv_subnet_constructor if conv else linear_subnet_constructor
        if len(c_internal) == 0:
            c_internal = [data_shape[0],]
        if len(c_internal) == 1:
            c_internal += c_internal

        if data_shape[0] >= 2 * min_split_size and max_splits != 0:
            self.leaf = False
            self.split_idx = data_shape[0] // 2
            self.upper = HierarchicalAffineCouplingTree((self.split_idx,) + data_shape[1:],
                                                        conv, subnet_constructor, c_internal[1:], clamp, max_splits-1, min_split_size)
            self.lower = HierarchicalAffineCouplingTree((data_shape[0] - self.split_idx,) + data_shape[1:],
                                                        conv, subnet_constructor, c_internal[1:], clamp, max_splits-1, min_split_size)
            self.s = subnet_constructor(self.split_idx, data_shape[0] - self.split_idx, c_internal[0])
            self.t = subnet_constructor(self.split_idx, data_shape[0] - self.split_idx, c_internal[0])

        else:
            self.leaf = True
            self.split_idx = data_shape[0] // 2
            self.s = subnet_constructor(self.split_idx, data_shape[0] - self.split_idx, c_internal[0])
            self.t = subnet_constructor(self.split_idx, data_shape[0] - self.split_idx, c_internal[0])

    def e(self, s):
        '''Soft-clamped exponential function'''
        return torch.exp(self.clamp * 0.636 * torch.atan(s))

    def log_e(self, s):
        '''Logarithm of the nonlinear function e'''
        return self.clamp * 0.636 * torch.atan(s)

    def forward(self, x, rev=False):
        if not self.leaf:
            x_upper, x_lower = torch.split(x, [self.split_idx, x.shape[1] - self.split_idx], dim=1)

            # Recursively run subtree transformations
            if not rev:
                x_upper, J_upper = self.upper.forward(x_upper, rev=rev)
                x_lower, J_lower = self.lower.forward(x_lower, rev=rev)

            # Compute own coupling transform and Jacobian
            s, t = self.s(x_upper), self.t(x_upper)
            if not rev:
                x_lower = self.e(s) * x_lower + t
                J = self.log_e(s)
            else:
                x_lower = (x_lower - t) / self.e(s)
                J = -self.log_e(s)

            # Reverse order of hierarchy during inverse pass
            if rev:
                x_upper, J_upper = self.upper.forward(x_upper, rev=rev)
                x_lower, J_lower = self.lower.forward(x_lower, rev=rev)

            x = torch.cat([x_upper, x_lower], dim=1)

            # Calculate block log Jacobian determinant
            J = torch.sum(J, dim=tuple(range(1, len(J.shape))))
            J = J_upper + J + J_lower

        else:
            # Compute coupling transform
            x_upper, x_lower = torch.split(x, [self.split_idx, x.shape[1] - self.split_idx], dim=1)
            s, t = self.s(x_upper), self.t(x_upper)
            if not rev:
                x_lower = self.e(s) * x_lower + t
                J = self.log_e(s)
            else:
                x_lower = (x_lower - t) / self.e(s)
                J = -self.log_e(s)
            x = torch.cat([x_upper, x_lower], dim=1)

            # Compute log determinant of triangular Jacobian
            J = torch.sum(J, dim=tuple(range(1, len(J.shape))))

        return x, J



class HierarchicalAffineCouplingBlock(nn.Module):
    '''TODO
    '''

    def __init__(self, dims_in, dims_c=[], conv=False, subnet_constructor=None, c_internal=[], clamp=4., max_splits=-1, min_split_size=2):
        super().__init__()
        self.tree = HierarchicalAffineCouplingTree(dims_in[0],
                                                   conv=conv,
                                                   subnet_constructor=subnet_constructor,
                                                   c_internal=c_internal,
                                                   clamp=clamp,
                                                   max_splits=max_splits,
                                                   min_split_size=min_split_size,
                                                   leaf_type=leaf_type)

    def forward(self, x, c=[], rev=False):
        x, self.jac = self.tree.forward(x[0], rev=rev)
        return [x]

    def jacobian(self, x, c=[], rev=False):
        return self.jac

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use one input."
        return input_dims
