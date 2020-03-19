import os, glob, json
import numpy as np
import torch
import torch.utils.data
import pickle
from numpy.random import rand, randn
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
from collections import defaultdict
from shapely import geometry as geo
from shapely.ops import nearest_points
from tqdm import tqdm

# np.seterr(divide='ignore', invalid='ignore')


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

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train)),
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test)),
                                              batch_size=batch_size,
                                              shuffle=True,
                                              drop_last=True)
    return train_loader, test_loader


def plot_dataset_example(model, limits=[-5,4,-4,5], n_samples=1000, seed=0):
    np.random.seed(seed)
    x = model.sample_prior(n_samples)
    x = model.unflatten_coeffs(x)
    points = model.trace_fourier_curves(x)

    fig = plt.figure(figsize=(15.3,3))
    axes = fig.subplots(1,5)

    for i in range(len(points)):
        axes[0].plot(points[i,:,0], points[i,:,1], c=(0,0,0,min(1,10/len(points))), zorder=1)
    axes[0].axvline(0, c='gray', ls=':', lw=.5, zorder=-1); axes[0].axhline(0, c='gray', ls=':', lw=.5, zorder=-1)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].axis(limits)

    for i in range(4):
        if model.name == 'plus-shape':
            coords = model.generate_plus_shape()
            axes[i+1].fill(coords[:,0], coords[:,1], fc=(1,1,1,0), ec=(1,0,0,.5), lw=2, zorder=-10)
            # axes[i+1].scatter(coords[:,0], coords[:,1], c=[(1,0,0,.25)], s=1, zorder=-10)
            points[i,:] = model.trace_fourier_curves(model.fourier_coeffs(coords, 25)[None,:,:])
        axes[i+1].plot(points[i,:,0], points[i,:,1], c=(0,0,0), lw=1, zorder=1)
        axes[i+1].axvline(0, c='gray', ls=':', lw=.5, zorder=-1); axes[i+1].axhline(0, c='gray', ls=':', lw=.5, zorder=-1)
        axes[i+1].set_xticks([]); axes[i+1].set_yticks([])
        axes[i+1].axis(limits)

    plt.subplots_adjust(left=.01, bottom=.01, right=.99, top=.99, wspace=.02, hspace=.01)
    plt.savefig(f'data/{model.name}_example.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.savefig(f'data/{model.name}_example.png', bbox_inches='tight', pad_inches=0.05, dpi=200)
    plt.show()


def plot_model_unconditional(model, c, model_inverse, limits=[-5,4,-4,5], n_samples=1000):
    z = torch.randn(n_samples, c.ndim_z).to(c.device)
    x = model_inverse(z)
    coeffs = c.data_model.unflatten_coeffs(x.data.cpu().numpy())
    points = c.data_model.trace_fourier_curves(coeffs)

    fig = plt.figure(num=c.suffix, figsize=(15.3,3))
    axes = fig.subplots(1,5)

    for i in range(len(points)):
        axes[0].plot(points[i,:,0], points[i,:,1], c=(0,0,0,min(1,10/len(points))), zorder=1)
    axes[0].axvline(0, c='gray', ls=':', lw=.5, zorder=-1); axes[0].axhline(0, c='gray', ls=':', lw=.5, zorder=-1)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].axis(limits)

    for i in range(4):
        axes[i+1].plot(points[i,:,0], points[i,:,1], c=(0,0,0), lw=1, zorder=1)
        axes[i+1].axvline(0, c='gray', ls=':', lw=.5, zorder=-1); axes[i+1].axhline(0, c='gray', ls=':', lw=.5, zorder=-1)
        axes[i+1].set_xticks([]); axes[i+1].set_yticks([])
        axes[i+1].axis(limits)

    plt.subplots_adjust(left=.01, bottom=.01, right=.99, top=.99, wspace=.02, hspace=.01)
    plt.savefig(f'data/{c.suffix}_example.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.savefig(f'data/{c.suffix}_example.png', bbox_inches='tight', pad_inches=0.05, dpi=200)
    plt.show()


def plot_model_conditional(model, c, model_inverse, y_target, limits=[-5,4,-4,5], n_samples=1000):
    if 'hint' in c.suffix:
        z = torch.randn(n_samples, c.ndim_x).cuda()
        y_target = torch.Tensor([y_target]*n_samples).view(n_samples,3).cuda()
        model_sample = model_inverse(y_target, z).data.cpu().numpy()
    else:
        z = torch.randn(n_samples, c.ndim_z).cuda()
        y_target = torch.Tensor([y_target]*n_samples).view(n_samples,3).cuda()
        model_sample = model_inverse(y_target, z).data.cpu().numpy()

    coeffs = c.data_model.unflatten_coeffs(model_sample)
    points = c.data_model.trace_fourier_curves(coeffs)

    fig = plt.figure(num=c.suffix, figsize=(15.3,3))
    axes = fig.subplots(1,5)

    for i in range(len(points)):
        axes[0].plot(points[i,:,0], points[i,:,1], c=(0,0,0,min(1,10/len(points))), zorder=1)
    axes[0].axvline(0, c='gray', ls=':', lw=.5, zorder=-1); axes[0].axhline(0, c='gray', ls=':', lw=.5, zorder=-1)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].axis(limits)

    for i in range(4):
        axes[i+1].plot(points[i,:,0], points[i,:,1], c=(0,0,0), lw=1, zorder=1)
        axes[i+1].axvline(0, c='gray', ls=':', lw=.5, zorder=-1); axes[i+1].axhline(0, c='gray', ls=':', lw=.5, zorder=-1)
        # Find largest diameter of the shape
        d = squareform(pdist(points[i]))
        max_idx = np.unravel_index(d.argmax(), d.shape)
        p0, p1 = points[i,max_idx[0]], points[i,max_idx[1]]
        angle = np.arctan2((p1-p0)[1], (p1-p0)[0])
        max_diameter = d[max_idx]
        # Find largest width orthogonal to diameter
        cos, sin = np.cos(angle), np.sin(angle)
        rotation = np.matrix([[cos, sin], [-sin, cos]])
        p_rotated = np.dot(rotation, points[i].T).T
        min_diameter = np.max(p_rotated[:,1]) - np.min(p_rotated[:,1])
        aspect_ratio = min_diameter / max_diameter
        # Make bounding boxes
        x_min, x_max = np.min(p_rotated[:,0]), np.max(p_rotated[:,0])
        y_min, y_max = np.min(p_rotated[:,1]), np.max(p_rotated[:,1])
        rect = np.array([[x_min,y_min], [x_min,y_max], [x_max,y_max], [x_max,y_min], [x_min,y_min]])
        cos, sin = np.cos(-angle), np.sin(-angle)
        rotation = np.matrix([[cos, sin], [-sin, cos]])
        rect = np.dot(rotation, rect.T).T
        target_rect = rect_with_given_aspect_and_angle(y_target[0,0].item(), y_target[0,2].item())
        target_rect = np.mean(rect[:-1,:], axis=0) + 0.5*max_diameter * target_rect
        axes[i+1].plot(target_rect[:,0], target_rect[:,1], c=(0,1,0,.2), lw=2, zorder=-2)
        axes[i+1].plot(rect[:,0], rect[:,1], c=(1,0,0,.2), lw=1, zorder=-1)
        # Circularity
        target_star = 0.95 * np.min(limits) * star_with_given_circularity(y_target[0,1].item(), n=36)
        axes[i+1].plot(target_star[:,0], target_star[:,1], c=(0,1,0,.2), lw=2, zorder=-2)
        shape = geo.Polygon(points[i])
        circularity = 4*np.pi * shape.area / shape.length**2
        star = 0.95 * np.min(limits) * star_with_given_circularity(circularity, n=36)
        axes[i+1].plot(star[:,0], star[:,1], c=(1,0,0,.2), lw=1, zorder=-1)

        axes[i+1].set_xticks([]); axes[i+1].set_yticks([])
        axes[i+1].axis(limits)

    plt.subplots_adjust(left=.01, bottom=.01, right=.99, top=.99, wspace=.02, hspace=.01)
    plt.savefig(f'data/{c.suffix}_example.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.savefig(f'data/{c.suffix}_example.png', bbox_inches='tight', pad_inches=0.05, dpi=200)
    plt.show()


def plot_model_conditional_abc(model, c, model_inverse, limits=[-5,4,-4,5], n_samples=1000, i=0):
    with open(f'abc/{c.data_model.name}/{i:05}.pkl', 'rb') as f:
        y_target, gt_sample, threshold = pickle.load(f)
    # y_target = c.vis_y_target
    if 'hint' in c.suffix:
        z = torch.randn(n_samples, c.ndim_x).cuda()
        y_target = torch.Tensor([y_target]*n_samples).view(n_samples,3).cuda()
        model_sample = model_inverse(y_target, z).data.cpu().numpy()
    else:
        z = torch.randn(n_samples, c.ndim_z).cuda()
        y_target = torch.Tensor([y_target]*n_samples).view(n_samples,3).cuda()
        model_sample = model_inverse(y_target, z).data.cpu().numpy()
    samples = [gt_sample[:n_samples,:], model_sample]
    # samples = [model_sample]

    fig = plt.figure(num=c.suffix, figsize=(6.2,3))
    axes = fig.subplots(1,2)

    for i, sample in enumerate(samples):
        coeffs = c.data_model.unflatten_coeffs(samples[i])
        points = c.data_model.trace_fourier_curves(coeffs)

        for j in range(len(points)):
            axes[i].plot(points[j,:,0], points[j,:,1], c=(0,0,0,min(1,10/len(points))), zorder=1)
        axes[i].axvline(0, c='gray', ls=':', lw=.5, zorder=-1); axes[i].axhline(0, c='gray', ls=':', lw=.5, zorder=-1)
        axes[i].set_xticks([]); axes[i].set_yticks([])
        axes[i].axis(limits)

    plt.subplots_adjust(left=.01, bottom=.01, right=.99, top=.99, wspace=.02, hspace=.01)
    # plt.savefig(f'data/{model.name}_example.pdf', bbox_inches='tight', pad_inches=0.05)
    # plt.savefig(f'data/{model.name}_example.png', bbox_inches='tight', pad_inches=0.05, dpi=200)
    plt.show()


def plot_fouriercurve_example():
    model = PlusShapeModel()
    with open(f'data/frog.json', 'r') as file:
        points = json.load(file)['points']
    points = np.array([[p['x'], p['y']] for p in points])
    points_dense = model.densify_polyline(points, 0.012)
    Ms = [1,2,3,5,10,20]
    coeffs = [model.fourier_coeffs(points, 2*i+1)[None,:,:] for i in Ms]
    curves = [model.trace_fourier_curves(c, 200)[0] for c in coeffs]

    fig = plt.figure(figsize=(9.5,3))
    axes = fig.subplots(1,3)

    axes[0].fill(points[:,0], points[:,1], fc=(0,0,0,.1), ec=(0,0,0,.5), lw=2, zorder=1)
    axes[1].plot(points[:,0], points[:,1], c=(1,0,0,.5), lw=1, zorder=1)
    axes[1].scatter(points_dense[:,0], points_dense[:,1], c=[(1,0,0)], s=1, zorder=1)
    axes[2].set_prop_cycle(plt.cycler('color', plt.cm.viridis(np.linspace(0.2,.9,len(Ms))[::-1])))
    for i in range(len(curves)):
        axes[2].plot(curves[i][:,0], curves[i][:,1], lw=1, zorder=1, label=2*Ms[i]+1)
    axes[2].legend(loc='upper center', title='# Fourier terms', ncol=3, fontsize=6)

    for i in range(3):
        axes[i].set_xticks([]); axes[i].set_yticks([])
        axes[i].axis([-.2,1.2,-.1,1.3])
    plt.subplots_adjust(left=.01, bottom=.01, right=.99, top=.99, wspace=.02, hspace=.01)
    plt.savefig(f'data/general_example.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.savefig(f'data/general_example.png', bbox_inches='tight', pad_inches=0.05, dpi=200)
    plt.show()


if __name__ == '__main__':
    pass

    # plot_dataset_example(FourierCurveModel(), limits=[-5,4,-4,5], seed=10)
    plot_dataset_example(PlusShapeModel(), limits=[-4,4,-4,4], seed=11)
    # plot_fouriercurve_example()


    # from configs.fourier_curve.conditional_hint_4_full import c, model, model_inverse
    # model.load_state_dict(torch.load('output/fourier-curve_conditional_hint-4-full.pt')['net'])
    # # from configs.fourier_curve.conditional_cinn_4 import c, model, model_inverse
    # # model.load_state_dict(torch.load('output/fourier-curve_conditional_cinn-4.pt')['net'])
    # plot_model_conditional_abc(model, c, model_inverse, i=10)


    # from configs.fourier_curve.conditional_hint_4_full import c, model, model_inverse
    # model.load_state_dict(torch.load('output/fourier-curve_conditional_hint-4-full.pt')['net'])
    # # from configs.fourier_curve.conditional_cinn_4 import c, model, model_inverse
    # # model.load_state_dict(torch.load('output/fourier-curve_conditional_cinn-4.pt')['net'])
    # plot_model_conditional(model, c, model_inverse, y_target=c.vis_y_target, limits=[-4,4,-4,4])

    # from configs.fourier_curve.unconditional_hint_2_full import c, model, model_inverse
    # model.load_state_dict(torch.load('output/fourier-curve_unconditional_hint-2-full.pt')['net'])
    # # from configs.fourier_curve.unconditional_inn_2 import c, model, model_inverse
    # # model.load_state_dict(torch.load('output/fourier-curve_unconditional_inn-2.pt')['net'])
    # plot_model_unconditional(model, c, model_inverse)


    # from configs.plus_shape.conditional_hint_4_full import c, model, model_inverse
    # model.load_state_dict(torch.load('output/plus-shape_conditional_hint-4-full.pt')['net'])
    # # from configs.plus_shape.conditional_cinn_4 import c, model, model_inverse
    # # model.load_state_dict(torch.load('output/plus-shape_conditional_cinn-4.pt')['net'])
    # plot_model_conditional(model, c, model_inverse, y_target=c.vis_y_target, limits=[-4,4,-4,4])

    # # from configs.plus_shape.unconditional_hint_4_full import c, model, model_inverse
    # # model.load_state_dict(torch.load('output/plus-shape_unconditional_hint-4-full.pt')['net'])
    # from configs.plus_shape.unconditional_inn_4 import c, model, model_inverse
    # model.load_state_dict(torch.load('output/plus-shape_unconditional_inn-4.pt')['net'])
    # plot_model_unconditional(model, c, model_inverse)
