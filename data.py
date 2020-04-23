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

# np.seterr(divide='ignore', invalid='ignore')


''' UCI data preparation adapted from https://github.com/LukasRinder/normalizing-flows
To use these data sets, download https://zenodo.org/record/1161203/files/data.tar.gz
and extract 'power', 'gas' and 'miniboone' into a directory named 'uci_data'
'''


# Helper classes and functions

class Data:
    def __init__(self, data):
        self.x = data.astype(np.float32)
        self.N = self.x.shape[0]

def star_with_given_circularity(circularity=1, n=7):
    # Spread 2*n points along unit circle
    angles = np.linspace(0, 2*np.pi, 2*n+1)
    a = angles[1]
    x = np.cos(angles)
    y = np.sin(angles)
    xy = np.stack([x,y]).T
    # Calculate inner radius for given circularity
    #    circularity = 4*np.pi * area / perimeter**2
    #    circularity = 4*np.pi * 2*n * (r * np.sin(a) / 2) / (2*n * np.sqrt(r*r + 1 - 2*r*np.cos(a)))**2
    #    circularity = 2*np.pi * r * np.sin(a)) / (2*n * (r*r + 1 - 2*r*np.cos(a)))
    #    r + 1/r = np.pi * np.sin(a) / (n * circularity) + 2*np.cos(a)
    c = np.pi * np.sin(a) / (n * circularity) + 2*np.cos(a)
    if c > 2:
        r = 0.5*(c - np.sqrt(c*c - 4))
        xy[::2,:] *= r
    return xy

def rect_with_given_aspect_and_angle(aspect_ratio, angle):
    xy = np.array([[-1,1], [-1,-1], [1,-1], [1,1], [-1,1]], dtype=float)
    xy[:,1] *= aspect_ratio
    rotation = np.matrix([[np.cos(-angle), np.sin(-angle)], [-np.sin(-angle), np.cos(-angle)]])
    xy = np.dot(rotation, xy.T).T
    return xy



# Actual data sets

class FourierCurveModel():

    n_parameters = 4*5 # must be uneven number times four
    n_observations = 3
    name = 'fourier-curve'

    def __init__(self):
        self.name = 'fourier-curve'
        self.coeffs_shape = (2, FourierCurveModel.n_parameters//4, 2)
        # Gaussian mixture for generating curve coefficients
        rng = np.random.RandomState(seed=123)
        self.n_components = 5
        self.component_weights = (.5 + rng.rand(self.n_components))
        self.component_weights /= np.sum(self.component_weights)
        self.mus = [.5 * rng.randn(*self.coeffs_shape) for i in range(self.n_components)]
        self.sigmas = [.1 + .2 * rng.rand(*self.coeffs_shape) for i in range(self.n_components)]

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

    def fourier_coeffs(self, points, n_coeffs=n_parameters//4):
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

    def sample_prior(self, n_samples, flat=True):
        samples = []
        for i in range(n_samples):
            c = np.random.choice(self.n_components, p=self.component_weights)
            sample = self.mus[c] + self.sigmas[c] * np.random.randn(*self.coeffs_shape)
            samples.append(sample.astype(np.float32).view(np.complex64))
        samples = np.stack(samples)
        if flat:
            samples = self.flatten_coeffs(samples)
        return samples

    def logprior(self, x):
        p = 0.0
        for i in range(self.n_components):
            p += self.component_weights[i] * np.exp(-0.5 * np.sum(((x.reshape(self.coeffs_shape) - self.mus[i]) / self.sigmas[i])**2)) / np.prod(self.sigmas[i])
        return np.log(p)

    def forward_process(self, x, noise=0.05):
        x = self.unflatten_coeffs(x)
        points = self.trace_fourier_curves(x)
        features = []
        for i in range(len(x)):
            # Find largest diameter of the shape
            d = squareform(pdist(points[i]))
            max_idx = np.unravel_index(d.argmax(), d.shape)
            p0, p1 = points[i,max_idx[0]], points[i,max_idx[1]]
            angle = np.arctan2((p1-p0)[1], (p1-p0)[0])
            max_diameter = d[max_idx]
            # Find largest width orthogonal to diameter
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.matrix([[c, s], [-s, c]])
            p_rotated = np.dot(rotation, points[i].T).T
            min_diameter = np.max(p_rotated[:,1]) - np.min(p_rotated[:,1])
            # Aspect ratio
            aspect_ratio = min_diameter / max_diameter
            # Circularity
            shape = geo.Polygon(points[i])
            circularity = 4*np.pi * shape.area / shape.length**2
            features.append((aspect_ratio, circularity, angle))
        features = np.array(features)
        return features + noise * randn(*features.shape)

    def loglikelihood(self, x, y, sigma_squared=0.05**2):
        if len(x.shape) <= 1:
            x = x[None,:]
            y = y[None,:]
        return -0.5 * np.sum((y - self.forward_process(x, noise=0))**2) / sigma_squared

    def logposterior(self, x, y, sigma_squared=0.05**2):
        return self.logprior(x) + self.loglikelihood(x, y, sigma_squared)

    def init_plot(self, y_target=None):
        return plt.figure(figsize=(7,7))

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
                        aspect_ratio, circularity, angle = y_target
                        # Visualize circularity
                        star = np.array((4,4)) + .5 * star_with_given_circularity(circularity)
                        plt.plot(star[:,0], star[:,1], c=(0,0,0,.25), lw=1)
                        # Visualize aspect ratio and angle
                        rect = np.array((4,2.5)) + .4 * rect_with_given_aspect_and_angle(aspect_ratio, angle)
                        plt.plot(rect[:,0], rect[:,1], c=(0,0,0,.25), lw=1)
                    # Find largest diameter of the shape
                    d = squareform(pdist(points[i]))
                    max_idx = np.unravel_index(d.argmax(), d.shape)
                    p0, p1 = points[i,max_idx[0]], points[i,max_idx[1]]
                    angle = np.arctan2((p1-p0)[1], (p1-p0)[0])
                    max_diameter = d[max_idx]
                    # Plot
                    d0, d1 = points[i,max_idx[0]], points[i,max_idx[1]]
                    plt.plot([d0[0], d1[0]], [d0[1], d1[1]], c=(0,1,0), ls='-', lw=1)
                    plt.scatter([d0[0], d1[0]], [d0[1], d1[1]], c=[(0,1,0)], s=3, zorder=10)
                    if y_target is not None:
                        # Find largest width orthogonal to diameter
                        c, s = np.cos(angle), np.sin(angle)
                        rotation = np.matrix([[c, s], [-s, c]])
                        p_rotated = np.dot(rotation, points[i].T).T
                        min_diameter = np.max(p_rotated[:,1]) - np.min(p_rotated[:,1])
                        # Aspect ratio & circularity
                        aspect_ratio = min_diameter / max_diameter
                        shape = geo.Polygon(points[i])
                        circularity = 4*np.pi * shape.area / shape.length**2
                        # Visualize circularity
                        star = np.array((4,4)) + .5 * star_with_given_circularity(circularity)
                        plt.plot(star[:,0], star[:,1], c=(0,1,0,.5), ls='-', lw=1)
                        # Visualize aspect ratio and angle
                        rect = np.array((4,2.5)) + .4 * rect_with_given_aspect_and_angle(aspect_ratio, angle)
                        plt.plot(rect[:,0], rect[:,1], c=(0,1,0,.5), ls='-', lw=1)
        plt.axis('equal')
        plt.axis([min(-5, points[:,:,0].min() - 1), max(5, points[:,:,0].max() + 1),
                  min(-5, points[:,:,1].min() - 1), max(5, points[:,:,1].max() + 1)])



class PlusShapeModel(FourierCurveModel):

    n_parameters = 4*25 # must be uneven number times four
    n_observations = 3
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
                             for t in np.linspace(0, 1, max(1, int(round(np.max(np.abs(end-start))/max_dist))))])
            all.append(dense)
        return np.concatenate(all)

    def generate_plus_shape(self):
        # Properties of x and y bar
        xlength = 3 + 2 * rand()
        ylength = 3 + 2 * rand()
        xwidth = .5 + 1.5 * rand()
        ywidth = .5 + 1.5 * rand()
        xshift = -1.5 + 3 * rand()
        yshift = -1.5 + 3 * rand()
        # Create bars and compute union
        xbar = geo.box(xshift - xlength/2, -xwidth/2, xshift + xlength/2, xwidth/2)
        ybar = geo.box(-ywidth/2, yshift - ylength/2, ywidth/2, yshift + ylength/2)
        both = xbar.union(ybar)
        coords = np.array(both.exterior.coords[:-1])
        # Add points inbetween, center, rotate and shift randomly
        coords = self.densify_polyline(coords)
        coords -= coords.mean(axis=0)
        angle = 0.5*np.pi * rand()
        rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        coords = np.dot(coords, rotation)
        coords += 0.5 * randn(1,2)
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



class Power:

    name = 'power'
    n_parameters = 6

    def __init__(self):

        trn, val, tst = self.load_data_normalised()

        self.trn = Data(trn)
        self.val = Data(val)
        self.tst = Data(tst)

        self.n_dims = self.trn.x.shape[1]

    def load_data(self):
        return np.load('uci_data/power/data.npy')

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

    def load_data_normalised(self):

        data_train, data_validate, data_test = self.load_data_split_with_noise()
        data = np.vstack((data_train, data_validate))
        mu = data.mean(axis=0)
        s = data.std(axis=0)
        data_train = (data_train - mu)/s
        data_validate = (data_validate - mu)/s
        data_test = (data_test - mu)/s

        return data_train, data_validate, data_test



class Gas:

    name = 'gas'
    n_parameters = 8

    def __init__(self):

        trn, val, tst = self.load_data_and_clean_and_split('uci_data/gas/ethylene_CO.pickle')

        self.trn = Data(trn)
        self.val = Data(val)
        self.tst = Data(tst)

        self.n_dims = self.trn.x.shape[1]

    def load_data(self, file):

        data = pd.read_pickle(file)
        data.drop("Meth", axis=1, inplace=True)
        data.drop("Eth", axis=1, inplace=True)
        data.drop("Time", axis=1, inplace=True)
        return data

    def get_correlation_numbers(self, data):

        C = data.corr()
        A = C > 0.98
        B = A.values.sum(axis=1)
        return B

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

    def load_data_and_clean_and_split(self, file):

        data = self.load_data_and_clean(file).values
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data_train = data[0:-N_test]
        N_validate = int(0.1*data_train.shape[0])
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]

        return data_train, data_validate, data_test



class Miniboone:

    name = 'miniboone'
    n_parameters = 42

    def __init__(self):

        trn, val, tst = self.load_data_normalised('uci_data/miniboone/data.npy')

        self.trn = Data(trn[:,0:-1])
        self.val = Data(val[:,0:-1])
        self.tst = Data(tst[:,0:-1])

        self.n_dims = self.trn.x.shape[1]

    def load_data(self, root_path):

        data = np.load(root_path)
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]

        return data_train, data_validate, data_test

    def load_data_normalised(self, root_path):

        data_train, data_validate, data_test = self.load_data(root_path)
        data = np.vstack((data_train, data_validate))
        mu = data.mean(axis=0)
        s = data.std(axis=0)
        data_train = (data_train - mu)/s
        data_validate = (data_validate - mu)/s
        data_test = (data_test - mu)/s

        return data_train, data_validate, data_test





def prepare_data_loaders(model, n_train, n_test, batch_size):
    try:
        x_train = np.load(f'data/{model.name}_x_train.npy')[:n_train,:]
        x_test = np.load(f'data/{model.name}_x_test.npy')[:n_test,:]
    except Exception as e:
        print(f'\nNot enough training data for model "{model.name}" found, generating {n_train + n_test} new samples...\n')
        x_train = model.sample_prior(n_train)
        np.save(f'data/{model.name}_x_train', x_train)
        x_test = model.sample_prior(n_test)
        np.save(f'data/{model.name}_x_test', x_test)
    try:
        y_train = np.load(f'data/{model.name}_y_train.npy')[:n_train,]
        y_test = np.load(f'data/{model.name}_y_test.npy')[:n_test,:]
    except Exception as e:
        print(f'\nNot enough training labels for model "{model.name}" found, running forward process on {n_train + n_test} samples...\n')
        y_train = []
        for i in range((n_train-1)//100000 + 1):
            print(f'Forward process chunk {i+1}...')
            y_train.append(model.forward_process(x_train[100000*i : min(n_train, 100000*(i+1)),:]))
        y_train = np.concatenate(y_train, axis=0)
        np.save(f'data/{model.name}_y_train', y_train)
        y_test = model.forward_process(x_test)
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

    # print('\n' + dataset_name, data.trn.x.shape, data.val.x.shape, data.tst.x.shape)
    train_loader = DataLoader(TensorDataset(torch.Tensor(data.trn.x), torch.zeros(len(data.trn.x),1)),
                              batch_size=batch_size, shuffle=shuffle, drop_last=True)
    test_loader  = DataLoader(TensorDataset(torch.Tensor(data.tst.x), torch.zeros(len(data.tst.x),1)),
                              batch_size=len(data.tst.x), shuffle=shuffle, drop_last=True)
    # print('train batches:', len(train_loader), '| test batches:', len(test_loader))

    return train_loader, test_loader



if __name__ == '__main__':
    pass

    # train, test = prepare_uci_loaders('power',     batch_size=1660)
    # train, test = prepare_uci_loaders('gas',       batch_size=853)
    # train, test = prepare_uci_loaders('miniboone', batch_size=30)
