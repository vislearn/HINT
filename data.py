import os, glob
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
from numpy.random import rand, randn
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
from collections import defaultdict
from shapely import geometry as geo
from shapely.ops import nearest_points
from tqdm import tqdm
from abc import ABCMeta, abstractmethod

# np.seterr(divide='ignore', invalid='ignore')



class FourierCurveModel(metaclass=ABCMeta):

    n_parameters = 4 # must be uneven number times four
    n_observations = 1
    name = ''

    @abstractmethod
    def __init__(self):
        pass

    def flatten_coeffs(self, coeffs):
        batch_size = coeffs.shape[0]
        coeffs = coeffs.reshape(batch_size, -1)
        return np.concatenate([coeffs.real, coeffs.imag], axis=1)

    def unflatten_coeffs(self, coeffs):
        batch_size = coeffs.shape[0]
        real, imag = np.split(coeffs, 2, axis=1)
        coeffs = real.astype(np.complex64)
        coeffs.imag = imag
        return coeffs.reshape(batch_size, 2, -1)

    def fourier_coeffs(self, points, n_coeffs):
        N = len(points) # Number of points
        M = n_coeffs//2
        M = min(N//2, M) # Number of positive/negative Fourier coefficients
        # Vectorized equation to compute Fourier coefficients
        ms = np.arange(-M, M+1)
        a = np.sum(points[:,:,None] * np.exp(-2*np.pi*1j*ms[None,None,:]*np.arange(N)[:,None,None]/N), axis=0) / N
        return a

    def trace_fourier_curves(self, coeffs, n_points=100):
        # Vectorized equation to compute points along the Fourier curve
        t = np.linspace(0, 1, n_points)
        ms = np.arange(-(coeffs.shape[-1]//2), coeffs.shape[-1]//2 + 1)
        tm = t[:,None] * ms[None,:]
        points = np.sum(coeffs[:,None,:,:] * np.exp(2*np.pi*1j*tm)[None,:,None,:], axis=-1).real
        return points

    @abstractmethod
    def sample_prior(self, n_samples, flat=True):
        pass

    @abstractmethod
    def sample_joint(self, n_samples, flat=True):
        pass

    def init_plot(self, y_target=None):
        return plt.figure(figsize=(7,7))

    @abstractmethod
    def update_plot(self, x, y_target=None, n_bold=3, show_forward=True):
        pass



class LensShapeModel(FourierCurveModel):

    n_parameters = 4*5 # 5 complex 2d Fourier coefficients
    n_observations = 2
    name = 'lens-shape'

    def __init__(self):
        self.name = 'lens-shape1'

    def generate_lens_shape(self):
        # First circle
        x0, y0, r0 = 0, 0, 1 + rand()
        p0 = geo.Point(x0, y0).buffer(r0)
        # Second circle
        r1 = 2*r0
        theta = 2*np.pi * rand() # Random angle
        d = 0.8 * (r0 + r1) # Distance of centers
        x1, y1 = x0 + d * np.sin(theta), y0 + d * np.cos(theta)
        p1 = geo.Point(x1, y1).buffer(r1)
        # Intersect
        shape = p0.intersection(p1)
        # Center with a little noise
        coords = np.array(shape.exterior.coords)
        coords -= coords.mean(axis=0) + 0.5 * randn(1,2)
        return coords

    def sample_prior(self, n_samples, flat=True):
        samples = []
        for i in range(n_samples):
            coords = self.generate_lens_shape()
            sample = self.fourier_coeffs(coords, n_coeffs=LensShapeModel.n_parameters//4)
            samples.append(sample)
        samples = np.stack(samples)
        if flat:
            samples = self.flatten_coeffs(samples)
        return samples

    def sample_joint(self, n_samples, flat=True):
        samples = []
        labels = []
        for i in tqdm(range(n_samples)):
            coords = self.generate_lens_shape()
            sample = self.fourier_coeffs(coords, n_coeffs=LensShapeModel.n_parameters//4)
            samples.append(sample[None,...])
            labels.append(self.forward_process(self.flatten_coeffs(samples[-1])))
        samples = np.concatenate(samples)
        labels = np.concatenate(labels)
        if flat:
            samples = self.flatten_coeffs(samples)
        return samples, labels

    def forward_process(self, x, noise=0.05):
        x = self.unflatten_coeffs(x)
        points = self.trace_fourier_curves(x)
        features = []
        for i in range(len(x)):
            # Find dominant angle and largest diameter of the shape
            d = squareform(pdist(points[i]))
            max_idx = np.unravel_index(d.argmax(), d.shape)
            p0, p1 = points[i,max_idx[0]], points[i,max_idx[1]]
            # features.append((angle, max_diameter))
            features.append(((p1-p0)[1], (p1-p0)[0]))
        features = np.array(features)
        return features + noise * randn(*features.shape)

    def update_plot(self, x, y_target=None, n_bold=3, show_forward=True):
        plt.gcf().clear()
        x = self.unflatten_coeffs(np.array(x))
        points = self.trace_fourier_curves(x)
        for i in range(len(points)):
            plt.plot(points[i,:,0], points[i,:,1], c=(0,0,0,min(1,10/len(points))))
            if i >= len(points) - n_bold:
                plt.plot(points[i,:,0], points[i,:,1], c=(0,0,0))
                if show_forward:
                    if y_target is not None:
                        diff_1, diff_0 = y_target
                        # Visualize angle and scale
                        # TODO
                    # Plot dominant angle and largest diameter of the shape
                    d = squareform(pdist(points[i]))
                    max_idx = np.unravel_index(d.argmax(), d.shape)
                    d0, d1 = points[i,max_idx[0]], points[i,max_idx[1]]
                    plt.plot([d0[0], d1[0]], [d0[1], d1[1]], c=(0,1,0), ls='-', lw=1)
                    plt.scatter([d0[0], d1[0]], [d0[1], d1[1]], c=[(0,1,0)], s=3, zorder=10)

        plt.axis('equal')
        plt.axis([min(-5, points[:,:,0].min() - 1), max(5, points[:,:,0].max() + 1),
                  min(-5, points[:,:,1].min() - 1), max(5, points[:,:,1].max() + 1)])



class PlusShapeModel(FourierCurveModel):

    n_parameters = 4*25 # 25 complex 2d Fourier coefficients
    n_observations = 4
    name = 'plus-shape'

    def __init__(self):
        self.name = 'plus-shape'

    def densify_polyline(self, coords, max_dist=0.2):
        # Add extra points between consecutive coordinates if they're too far apart
        all = []
        for i in range(len(coords)):
            start = coords[(i+1)%len(coords),:]
            end = coords[i,:]
            dense = np.array([t * start + (1-t) * end
                             for t in np.linspace(0, 1, max(1, int(round(
                                np.max(np.abs(end-start))/max_dist))))])
            all.append(dense)
        return np.concatenate(all)

    def generate_plus_shape(self, forward=False, target=None):
        # Properties of x and y bar
        xlength = 3 + 2 * rand()
        ylength = 3 + 2 * rand()
        if target is None:
            xwidth = .5 + 1.5 * rand()
            ywidth = .5 + 1.5 * rand()
        else:
            if target[3] >= 1:
                xwidth = target[3]*.5 + (2 - target[3]*0.5) * rand()
            else:
                xwidth = 0.5 + (2*target[3] - 0.5) * rand()
            ywidth = xwidth/target[3]
        xshift = -1.5 + 3 * rand()
        yshift = -1.5 + 3 * rand()
        center = np.array([0.0, 0.0])
        # Create bars and compute union
        xbar = geo.box(xshift - xlength/2, -xwidth/2, xshift + xlength/2, xwidth/2)
        ybar = geo.box(-ywidth/2, yshift - ylength/2, ywidth/2, yshift + ylength/2)
        both = xbar.union(ybar)
        coords = np.array(both.exterior.coords[:-1])
        # Add points inbetween, center, rotate and shift randomly
        coords = self.densify_polyline(coords)
        center -= coords.mean(axis=0)
        coords -= coords.mean(axis=0)
        if target is None:
            angle = 0.5*np.pi * rand()
        else:
            angle = target[2]
        rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        coords = np.dot(coords, rotation)
        center = np.dot(center, rotation)
        offset = 0.5 * randn(1,2)
        coords += offset
        center += offset[0,:]

        if forward:
            return coords, np.array([center[0], center[1], angle, xwidth/ywidth])
        else:
            return coords

    def sample_prior(self, n_samples, flat=True):
        samples = []
        for i in range(n_samples):
            coords = self.generate_plus_shape()
            sample = self.fourier_coeffs(coords, n_coeffs=PlusShapeModel.n_parameters//4)
            samples.append(sample)
        samples = np.stack(samples)
        if flat:
            samples = self.flatten_coeffs(samples)
        return samples

    def sample_joint(self, n_samples, flat=True):
        samples = []
        labels = []
        for i in tqdm(range(n_samples)):
            coords, label = self.generate_plus_shape(forward=True)
            sample = self.fourier_coeffs(coords, n_coeffs=PlusShapeModel.n_parameters//4)
            samples.append(sample)
            labels.append(label)
        samples = np.stack(samples)
        labels = np.stack(labels)
        if flat:
            samples = self.flatten_coeffs(samples)
        return samples, labels

    def update_plot(self, x, y_target=None, n_bold=3, show_forward=True):
        plt.gcf().clear()
        x = self.unflatten_coeffs(np.array(x))
        points = self.trace_fourier_curves(x)
        for i in range(len(points)):
            plt.plot(points[i,:,0], points[i,:,1], c=(0,0,0,min(1,10/len(points))), zorder=1)
            plt.axvline(0, c='gray', ls=':', lw=.5, zorder=-1)
            plt.axhline(0, c='gray', ls=':', lw=.5, zorder=-1)
            if i >= len(points) - n_bold:
                plt.plot(points[i,:,0], points[i,:,1], c=(0,0,0))
                if show_forward:
                    if y_target is not None:
                        center_x, center_y, angle, ratio = y_target
                        plt.plot([center_x - 100*np.cos(angle), center_x + 100*np.cos(angle)],
                            [center_y - 100*np.sin(angle), center_y + 100*np.sin(angle)],
                            lw=30, color=(0,1,0,0.1), zorder=-10)
                        plt.plot([center_x + 100*np.sin(angle), center_x - 100*np.sin(angle)],
                            [center_y - 100*np.cos(angle), center_y + 100*np.cos(angle)],
                            lw=30/ratio, color=(0,1,0,0.1), zorder=-10)

        plt.axis('equal')
        plt.axis([min(-5, points[:,:,0].min() - 1), max(5, points[:,:,0].max() + 1),
                  min(-5, points[:,:,1].min() - 1), max(5, points[:,:,1].max() + 1)])



# UCI data preparation
# Adapted from https://github.com/LukasRinder/normalizing-flows

class Data:
    def __init__(self, data):
        self.x = data.astype(np.float32)
        self.N = self.x.shape[0]


class Power:
    name = 'power'
    n_parameters = 6

    def __init__(self):
        trn, val, tst = Power.load_data_normalised()

        self.trn = Data(trn)
        self.val = Data(val)
        self.tst = Data(tst)

        self.n_dims = self.trn.x.shape[1]

    @classmethod
    def load_data(self):
        return np.load('uci_data/power/data.npy')

    @classmethod
    def load_data_split_with_noise(self):
        rng = np.random.RandomState(42)

        data = self.load_data()
        rng.shuffle(data)
        N = data.shape[0]

        data = np.delete(data, 3, axis=1)
        data = np.delete(data, 1, axis=1)
        # global_intensity_noise = 0.1*rng.rand(N, 1)
        voltage_noise = 0.01*rng.rand(N, 1)
        # grp_noise = 0.001*rng.rand(N, 1)
        gap_noise = 0.001*rng.rand(N, 1)
        sm_noise = rng.rand(N, 3)
        time_noise = np.zeros((N, 1))
        noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
        data = data + noise

        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]

        return data_train, data_validate, data_test

    @classmethod
    def load_data_normalised(self):
        data_train, data_validate, data_test = self.load_data_split_with_noise()
        data = np.vstack((data_train, data_validate))
        mu = data.mean(axis=0)
        s = data.std(axis=0)
        data_train = (data_train - mu)/s
        data_validate = (data_validate - mu)/s
        data_test = (data_test - mu)/s

        return data_train, data_validate, data_test

    @classmethod
    def mean_and_std(self):
        data_train, data_validate, data_test = self.load_data_split_with_noise()
        data = np.vstack((data_train, data_validate))
        return data.mean(axis=0), data.std(axis=0)


class Gas:
    name = 'gas'
    n_parameters = 8

    def __init__(self):
        trn, val, tst = Gas.load_data_and_clean_and_split('uci_data/gas/ethylene_CO.pickle')

        self.trn = Data(trn)
        self.val = Data(val)
        self.tst = Data(tst)

        self.n_dims = self.trn.x.shape[1]

    @classmethod
    def load_data(self, file):
        data = pd.read_pickle(file)
        data.drop("Meth", axis=1, inplace=True)
        data.drop("Eth", axis=1, inplace=True)
        data.drop("Time", axis=1, inplace=True)
        return data

    @classmethod
    def get_correlation_numbers(self, data):
        C = data.corr()
        A = C > 0.98
        B = A.values.sum(axis=1)
        return B

    @classmethod
    def load_data_and_clean(self, file):
        data = self.load_data(file)
        B = self.get_correlation_numbers(data)

        while np.any(B > 1):
            col_to_remove = np.where(B > 1)[0][0]
            col_name = data.columns[col_to_remove]
            data.drop(col_name, axis=1, inplace=True)
            B = self.get_correlation_numbers(data)

        data = (data-data.mean())/data.std()
        return data

    @classmethod
    def load_data_and_clean_and_split(self, file):
        data = self.load_data_and_clean(file).values
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data_train = data[0:-N_test]
        N_validate = int(0.1*data_train.shape[0])
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]

        return data_train, data_validate, data_test

    @classmethod
    def mean_and_std(self, file='uci_data/gas/ethylene_CO.pickle'):
        data = self.load_data(file)
        B = self.get_correlation_numbers(data)

        while np.any(B > 1):
            col_to_remove = np.where(B > 1)[0][0]
            col_name = data.columns[col_to_remove]
            data.drop(col_name, axis=1, inplace=True)
            B = self.get_correlation_numbers(data)

        return data.mean(), data.std()


class Miniboone:
    name = 'miniboone'
    n_parameters = 42

    def __init__(self):
        trn, val, tst = Miniboone.load_data_normalised('uci_data/miniboone/data.npy')

        self.trn = Data(trn[:,0:-1])
        self.val = Data(val[:,0:-1])
        self.tst = Data(tst[:,0:-1])

        self.n_dims = self.trn.x.shape[1]

    @classmethod
    def load_data(self, root_path):
        data = np.load(root_path)
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]

        return data_train, data_validate, data_test

    @classmethod
    def load_data_normalised(self, root_path):
        data_train, data_validate, data_test = self.load_data(root_path)
        data = np.vstack((data_train, data_validate))
        mu = data.mean(axis=0)
        s = data.std(axis=0)
        data_train = (data_train - mu)/s
        data_validate = (data_validate - mu)/s
        data_test = (data_test - mu)/s

        return data_train, data_validate, data_test

    @classmethod
    def mean_and_std(self):
        data_train, data_validate, data_test = self.load_data('uci_data/miniboone/data.npy')
        data = np.vstack((data_train, data_validate))
        return data.mean(axis=0), data.std(axis=0)



def prepare_data_loaders(model, n_train, n_test, batch_size):
    try:
        x_train = np.load(f'data/{model.name}_x_train.npy')[:n_train,:]
        y_train = np.load(f'data/{model.name}_y_train.npy')[:n_train,]
    except Exception as e:
        print(f'\nNot enough training data for model "{model.name}" found, generating {n_train} new training samples...')
        x_train, y_train = model.sample_joint(n_train)
        np.save(f'data/{model.name}_x_train', x_train)
        np.save(f'data/{model.name}_y_train', y_train)
    try:
        x_test = np.load(f'data/{model.name}_x_test.npy')[:n_test,:]
        y_test = np.load(f'data/{model.name}_y_test.npy')[:n_test,:]
    except Exception as e:
        print(f'\nNot enough test data for model "{model.name}" found, generating {n_test} new test samples...')
        x_test, y_test = model.sample_joint(n_test)
        np.save(f'data/{model.name}_x_test', x_test)
        np.save(f'data/{model.name}_y_test', y_test)

    train_loader = DataLoader(TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train)),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader =  DataLoader(TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test)),
                              batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader



def prepare_uci_loaders(dataset_name="power", batch_size=1000, shuffle=True):
    if dataset_name == "power":
        data = Power()
    elif dataset_name == "gas":
        data = Gas()
    elif dataset_name == "miniboone":
        data = Miniboone()
    else:
        raise ValueError("Dataset not known.")

    train_loader = DataLoader(TensorDataset(torch.Tensor(data.trn.x), torch.zeros(len(data.trn.x),1)),
                              batch_size=batch_size, shuffle=shuffle, drop_last=True)
    test_loader  = DataLoader(TensorDataset(torch.Tensor(data.tst.x), torch.zeros(len(data.tst.x),1)),
                              batch_size=len(data.tst.x), shuffle=shuffle, drop_last=True)

    return train_loader, test_loader



if __name__ == '__main__':
    pass

    # prepare_data_loaders(LensShapeModel(), 1000000, 100000, 10000)
    # prepare_data_loaders(PlusShapeModel(), 1000000, 100000, 10000)
