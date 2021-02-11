import json
import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
from shapely import geometry as geo
from shapely.ops import nearest_points

from data import *
from best_shape_fit import *



# From https://stackoverflow.com/a/42972469/12939023
from matplotlib.lines import Line2D
class LineDataUnits(Line2D):
    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop("linewidth", 1)
        super().__init__(*args, **kwargs)
        self._lw_data = _lw_data

    def _get_lw(self):
        if self.axes is not None:
            ppd = 72./self.axes.figure.dpi
            trans = self.axes.transData.transform
            return ((trans((1, self._lw_data))-trans((0, 0)))*ppd)[1]
        else:
            return 1

    def _set_lw(self, lw):
        self._lw_data = lw

    _linewidth = property(_get_lw, _set_lw)




def plot_dataset_example(model, limits=[-5,4,-4,5], n_samples=10000, seed=0):
    np.random.seed(seed)
    x = model.sample_prior(n_samples, flat=True)
    # x = model.sample_prior(n_samples, flat=False).reshape(n_samples, -1)

    fig = plt.figure(figsize=(15.3,3))
    axes = fig.subplots(1,5)

    for i in range(4):
        if model.name == 'plus-shape':
            coords = model.generate_plus_shape()
            axes[i].fill(coords[:,0], coords[:,1], fc=(1,1,1,0), ec=(1,0,0,.25), lw=2, zorder=-10)
            points = model.trace_fourier_curves(model.fourier_coeffs(coords, 25)[None,:,:])[0]
        if model.name == 'lens-shape':
            coords = model.generate_lens_shape()
            axes[i].fill(coords[:,0], coords[:,1], fc=(1,1,1,0), ec=(1,0,0,.25), lw=2, zorder=-10)
            points = model.trace_fourier_curves(model.fourier_coeffs(coords, 5)[None,:,:])[0]
        axes[i].plot(points[:,0], points[:,1], c=(0,0,0), lw=1, zorder=1)
        axes[i].axvline(0, c='gray', ls=':', lw=.5, zorder=-1)
        axes[i].axhline(0, c='gray', ls=':', lw=.5, zorder=-1)
        axes[i].set_xticks([]); axes[i].set_yticks([])
        axes[i].axis(limits)

    corr = np.corrcoef(x.T)
    # corr = np.corrcoef(x.T).imag
    np.save(f'data/{model.name}_corr.npy', corr)
    axes[4].imshow(corr, cmap='RdBu', interpolation='nearest')
    axes[4].set_yticks([]); axes[4].set_xticks([])

    plt.subplots_adjust(left=.01, bottom=.01, right=.99, top=.99, wspace=.02, hspace=.01)
    plt.savefig(f'data/{model.name}_example.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.savefig(f'data/{model.name}_example.png', bbox_inches='tight', pad_inches=0.05, dpi=200)
    plt.show()



def show_parameter_sensitivity(limits=[-4,4,-4,4], n_samples=5, seed=0):
    model = PlusShapeModel()
    np.random.seed(seed)
    coords_base = model.generate_plus_shape()

    fig = plt.figure(figsize=(9, 2*n_samples))
    axes = fig.subplots(n_samples, 5)

    for i in range(n_samples):
        coords = np.array(coords_base)
        axes[i][0].fill(coords[:,0], coords[:,1], fc=(1,1,1,0), ec=(1,0,0,.25), lw=2, zorder=-10)
        coeffs = model.fourier_coeffs(coords, 25)[None,:,:]
        for j in range(5):
            points = model.trace_fourier_curves(coeffs)[0]
            axes[i][j].plot(points[:,0], points[:,1], c=(0,0,0), lw=1, zorder=1)
            axes[i][j].axvline(0, c='gray', ls=':', lw=.5, zorder=-1)
            axes[i][j].axhline(0, c='gray', ls=':', lw=.5, zorder=-1)
            axes[i][j].set_xticks([]); axes[i][j].set_yticks([])
            axes[i][j].axis(limits)
            coeffs[0, i%2, 18+3*i//2] += 0.1 * ((i+1)%2) + 0.1j * (i%2)

    plt.subplots_adjust(left=.01, bottom=.01, right=.99, top=.99, wspace=0, hspace=.1)
    plt.savefig(f'data/parameter_sensitivity.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.savefig(f'data/parameter_sensitivity.png', bbox_inches='tight', pad_inches=0.05, dpi=200)
    plt.show()



def plot_model_unconditional(c, limits=[-4,4,-4,4], n_samples=10000):
    z = torch.randn(n_samples, c.ndim_z).to(c.device)
    x = c.model_inverse(z).detach().cpu().numpy()
    coeffs = c.data_model.unflatten_coeffs(x)
    points = c.data_model.trace_fourier_curves(coeffs)

    fig = plt.figure(num=c.suffix, figsize=(15.3,3))
    axes = fig.subplots(1,5)

    for i in range(4):
        axes[i].plot(points[i,:,0], points[i,:,1], c=(0,0,0), lw=1, zorder=1)
        axes[i].axvline(0, c='gray', ls=':', lw=.5, zorder=-1); axes[i].axhline(0, c='gray', ls=':', lw=.5, zorder=-1)
        axes[i].set_xticks([]); axes[i].set_yticks([])
        axes[i].axis('equal'); axes[i].axis(limits)
        if c.data_model.name == 'lens-shape':
            fit_params = fit_lens_shape_to_points(torch.tensor(points[i]).float(), verbose=False)
            fit_curve = lens_points_from_params(get_lens_prototype(), fit_params).detach().cpu().numpy()
            axes[i].plot(fit_curve[:,0], fit_curve[:,1], c=(1,0,0,.25), lw=2, zorder=-10)
        if c.data_model.name == 'plus-shape':
            fit_params = fit_plus_shape_to_points(torch.tensor(points[i]).float(), verbose=False)
            segments = plus_segments_from_params(fit_params).cpu().numpy()
            for segment in segments:
                axes[i].plot(segment[:,0], segment[:,1], c=(1,0,0,.25), lw=2, zorder=-10)

    corr = np.corrcoef(x.T)
    corr_true = np.load(f'data/{c.data_model.name}_corr.npy')
    corr_diff = np.abs(corr - corr_true)
    # print(np.nanmin(corr_diff), np.nanmax(corr_diff))
    axes[4].imshow(corr_diff, cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
    axes[4].set_yticks([]); axes[4].set_xticks([])

    plt.subplots_adjust(left=.01, bottom=.01, right=.99, top=.99, wspace=.02, hspace=.01)
    plt.savefig(f'data/{c.suffix}_example.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.savefig(f'data/{c.suffix}_example.png', bbox_inches='tight', pad_inches=0.05, dpi=200)
    plt.show()



def plot_model_conditional(c, limits=[-4,4,-4,4], n_samples=4000, diverse=False):
    z = torch.randn(n_samples, c.ndim_z if ('inn' in c.suffix) else c.ndim_x).cuda()
    if diverse:
        y_target = torch.cat([torch.rand(n_samples,2) * 2 - 1, torch.rand(n_samples,1) * .5 * np.pi, (torch.rand(n_samples,1) * 2 + 1).pow(torch.randn(n_samples,1).sign())], dim=1).cuda()
    else:
        y_target = torch.Tensor([c.vis_y_target]*n_samples).view(n_samples, c.data_model.n_observations).cuda()
    x = c.model_inverse(y_target, z).data.cpu().numpy()

    coeffs = c.data_model.unflatten_coeffs(x[:4])
    points = c.data_model.trace_fourier_curves(coeffs)

    fig = plt.figure(num=c.suffix, figsize=(15.3,3))
    axes = fig.subplots(1,5)

    for i in range(4):
        axes[i].plot(points[i,:,0], points[i,:,1], c=(0,0,0), lw=1, zorder=1)
        axes[i].axvline(0, c='gray', ls=':', lw=.5, zorder=-1); axes[i].axhline(0, c='gray', ls=':', lw=.5, zorder=-1)
        axes[i].set_xticks([]); axes[i].set_yticks([])
        axes[i].axis('equal'); axes[i].axis(limits)
        if c.data_model.name == 'lens-shape':
            # Plot dominant angle and largest diameter of the shape
            d = squareform(pdist(points[i]))
            max_idx = np.unravel_index(d.argmax(), d.shape)
            d0, d1 = points[i,max_idx[0]], points[i,max_idx[1]]
            axes[i].plot([d0[0], d1[0]], [d0[1], d1[1]], c=(0,1,0), ls=':', lw=3)
            axes[i].scatter([d0[0], d1[0]], [d0[1], d1[1]], c=[(0,1,0)], s=3, zorder=10)
            # Show correct angle/diameter
            p0 = (d0 + d1)/2 + np.array(c.vis_y_target)[::-1]/2
            p1 = (d0 + d1)/2 - np.array(c.vis_y_target)[::-1]/2
            axes[i].plot([p0[0], p1[0]], [p0[1], p1[1]], c=(1,0,0,.25), ls='-', lw=3, zorder=-11)
            axes[i].scatter([p0[0], p1[0]], [p0[1], p1[1]], c=[(1,0,0,.25)], s=5, zorder=-10)
            # fit_params = fit_lens_shape_to_points(torch.tensor(points).float(), verbose=False)
            # fit_curve = lens_points_from_params(get_lens_prototype(), fit_params).detach().cpu().numpy()
            # axes[i].plot(fit_curve[:,0], fit_curve[:,1], c=(1,0,0,.25), lw=2, zorder=-10)
        if c.data_model.name == 'plus-shape':
            # Fit proper Plus shape
            fit_params = fit_plus_shape_to_points(torch.tensor(points[i]).float(), verbose=False)
            segments = plus_segments_from_params(fit_params).cpu().numpy()
            for segment in segments:
                axes[i].plot(segment[:,0], segment[:,1], c=(1,0,0,.25), lw=2, zorder=-10)
            # Visualize condition
            center_x, center_y, angle, ratio = [y_target[i][j].item() for j in range(4)]
            xwidth = fit_params[2].item()
            ywidth = fit_params[3].item()
            width = max(xwidth, ywidth) if ratio > 1 else min(xwidth, ywidth)
            line = LineDataUnits([center_x - 100*np.cos(angle), center_x + 100*np.cos(angle)], [center_y - 100*np.sin(angle), center_y + 100*np.sin(angle)], linewidth=width, color=(.2,1,.5,0.1), zorder=-10)
            axes[i].add_line(line)
            line = LineDataUnits([center_x + 100*np.sin(angle), center_x - 100*np.sin(angle)], [center_y - 100*np.cos(angle), center_y + 100*np.cos(angle)], linewidth=width/ratio, color=(.2,1,.5,0.1), zorder=-10)
            axes[i].add_line(line)

    corr = np.corrcoef(x.T)
    corr_true = np.load(f'data/{c.data_model.name}_corr_conditional.npy')
    corr_diff = np.abs(corr - corr_true)
    # print(np.nanmin(corr_diff), np.nanmax(corr_diff))
    axes[4].imshow(corr_diff, cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
    axes[4].set_yticks([]); axes[4].set_xticks([])

    plt.subplots_adjust(left=.01, bottom=.01, right=.99, top=.99, wspace=.02, hspace=.01)
    suffix = '_diverse' if diverse else ''
    plt.savefig(f'data/{c.suffix}_example{suffix}.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.savefig(f'data/{c.suffix}_example{suffix}.png', bbox_inches='tight', pad_inches=0.05, dpi=200)
    plt.show()



def plot_model_conditional_abc(model, c, model_inverse, limits=[-5,4,-4,5], n_samples=1000, i=0):
    with open(f'abc/{c.data_model.name}/{i:05}.pkl', 'rb') as f:
        y_target, gt_sample, threshold = pickle.load(f)
    # y_target = c.vis_y_target
    if 'hint' in c.suffix:
        z = torch.randn(n_samples, c.ndim_x).cuda()
        y_target = torch.Tensor([y_target]*n_samples).view(n_samples,3).cuda()
        x = model_inverse(y_target, z).data.cpu().numpy()
    else:
        z = torch.randn(n_samples, c.ndim_z).cuda()
        y_target = torch.Tensor([y_target]*n_samples).view(n_samples,3).cuda()
        x = model_inverse(y_target, z).data.cpu().numpy()
    samples = [gt_sample[:n_samples,:], x]
    # samples = [x]

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
    Ms = [1,3,10,20] # [1,2,3,5,10,20]
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
    axes[2].legend(loc='upper center', title='# Fourier terms', ncol=4, fontsize=9)

    for i in range(3):
        axes[i].set_xticks([]); axes[i].set_yticks([])
        axes[i].axis([-.2,1.2,-.1,1.3])
    plt.subplots_adjust(left=.01, bottom=.01, right=.99, top=.99, wspace=.02, hspace=.01)
    plt.savefig(f'data/general_example.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.savefig(f'data/general_example.png', bbox_inches='tight', pad_inches=0.05, dpi=200)
    plt.show()


def metrics_illustration():
    # Create example shapes
    model = PlusShapeModel()
    with open(f'data/frog.json', 'r') as file:
        points = json.load(file)['points']
    points = np.array([[p['x'], p['y']] for p in points])
    points_dense = model.densify_polyline(points, 0.012)
    Ms = [4,30]
    coeffs = [model.fourier_coeffs(points, 2*i+1)[None,:,:] for i in Ms]
    curves = [model.trace_fourier_curves(c, 200)[0] for c in coeffs]
    rough = geo.Polygon(curves[0])
    fine = geo.Polygon(curves[1])

    fig = plt.figure(figsize=(10,5))
    axes = fig.subplots(1,2)

    # Plot IoU
    intersection = np.array(rough.intersection(fine).exterior.coords)
    union = np.array(rough.union(fine).exterior.coords)

    import matplotlib as mpl
    mpl.rcParams['hatch.linewidth'] = 3
    axes[0].fill(union[:,0], union[:,1], fc='#96BF0D', ec=(0,0,0), lw=2, zorder=1)
    axes[0].fill(intersection[:,0], intersection[:,1], fc='#E37238', ec='#96BF0D', hatch='///', lw=0, zorder=2)
    axes[0].plot(intersection[:,0], intersection[:,1], color=(0,0,0), lw=2, zorder=3)

    # Plot Hausdorff distance
    axes[1].plot(curves[0][:,0], curves[0][:,1], color='#E37238', lw=3, zorder=1)
    axes[1].plot(curves[1][:,0], curves[1][:,1], color='#96BF0D', lw=3, zorder=1)

    axes[1].scatter(curves[0][:,0], curves[0][:,1], color='#464646', s=4, zorder=3)
    axes[1].scatter(curves[1][:,0], curves[1][:,1], color='#464646', s=4, zorder=3)

    diffs = curves[0][None,:,:] - curves[1][:,None,:]
    dists = np.sqrt(np.sum(diffs*diffs, axis=-1))
    minima_0 = np.argmin(dists, axis=0)
    minima_1 = np.argmin(dists, axis=1)
    for i,j in enumerate(minima_0):
        axes[1].plot([curves[0][i,0], curves[1][j,0]], [curves[0][i,1], curves[1][j,1]], color='#464646', lw=1, zorder=5)
    for i,j in enumerate(minima_1):
        axes[1].plot([curves[0][j,0], curves[1][i,0]], [curves[0][j,1], curves[1][i,1]], color='#464646', lw=1, zorder=5)

    # Make plots pretty
    for i in range(2):
        axes[i].set_xticks([]); axes[i].set_yticks([])
        axes[i].axis([-.2,1.2,-.1,1.3])
        axes[i].set_frame_on(False)
        axes[i].axis('equal')
    plt.subplots_adjust(left=.01, bottom=.01, right=.99, top=.99, wspace=.02, hspace=.01)
    plt.show()



if __name__ == '__main__':
    pass

    # plot_dataset_example(FourierCurveModel(), limits=[-5,4,-4,5], seed=10)
    # plot_dataset_example(PlusShapeModel(), limits=[-4,4,-4,4], seed=8)
    # plot_dataset_example(LensShapeModel(), limits=[-2.5,2.5,-2.5,2.5], seed=1)
    # show_parameter_sensitivity()
    plot_fouriercurve_example()
    # metrics_illustration()


    # # from configs.lens_shape.unconditional_hint_1_full import c
    # # c.model.load_state_dict(torch.load('results/lens_shape-unconditional_hint_1_full_0.pt')['net'])
    # # from configs.lens_shape.unconditional_hint_2_full import c
    # # c.model.load_state_dict(torch.load('results/lens_shape-unconditional_hint_2_full_0.pt')['net'])
    # # from configs.lens_shape.unconditional_inn_1 import c
    # # c.model.load_state_dict(torch.load('results/lens_shape-unconditional_inn_1_0.pt')['net'])
    # from configs.lens_shape.unconditional_inn_2 import c
    # c.model.load_state_dict(torch.load('results/lens_shape-unconditional_inn_2_0.pt')['net'])
    # plot_model_unconditional(c, limits=[-3.5,3.5,-3.5,3.5])


    # # from configs.plus_shape.unconditional_hint_4_full import c
    # # c.model.load_state_dict(torch.load('results/plus_shape-unconditional_hint_4_full_0.pt')['net'])
    # # from configs.plus_shape.unconditional_hint_8_full import c
    # # c.model.load_state_dict(torch.load('results/plus_shape-unconditional_hint_8_full_0.pt')['net'])
    # # from configs.plus_shape.unconditional_inn_4_Q import c
    # # c.model.load_state_dict(torch.load('results/plus_shape-unconditional_inn_4_Q_0.pt')['net'])
    # # from configs.plus_shape.unconditional_inn_4 import c
    # # c.model.load_state_dict(torch.load('results/plus_shape-unconditional_inn_4_0.pt')['net'])
    # from configs.plus_shape.unconditional_inn_8 import c
    # c.model.load_state_dict(torch.load('results/plus_shape-unconditional_inn_8_0.pt')['net'])
    # plot_model_unconditional(c, limits=[-4,4,-4,4])


    # from configs.lens_shape.conditional_hint_1_full import c
    # c.model.load_state_dict(torch.load('results/lens_shape-conditional_hint_1_full_0.pt')['net'])
    # # from configs.lens_shape.conditional_hint_4_full import c
    # # c.model.load_state_dict(torch.load('results/lens_shape-conditional_hint_4_full_0.pt')['net'])
    # # from configs.lens_shape.conditional_cinn_1 import c
    # # c.model.load_state_dict(torch.load('results/lens_shape-conditional_cinn_1_0.pt')['net'])
    # # from configs.lens_shape.conditional_cinn_4 import c
    # # c.model.load_state_dict(torch.load('results/lens_shape-conditional_cinn_4_0.pt')['net'])
    # plot_model_conditional(c, limits=[-2.5,2.5,-2.5,2.5])


    # # from configs.plus_shape.conditional_hint_4_full import c
    # # c.model.load_state_dict(torch.load('results/plus_shape-conditional_hint_4_full_0.pt')['net'])
    # from configs.plus_shape.conditional_hint_8_full import c
    # c.model.load_state_dict(torch.load('results/plus_shape-conditional_hint_8_full_0.pt')['net'])
    # # from configs.plus_shape.conditional_cinn_4 import c
    # # c.model.load_state_dict(torch.load('results/plus_shape-conditional_cinn_4_0.pt')['net'])
    # # from configs.plus_shape.conditional_cinn_8 import c
    # # c.model.load_state_dict(torch.load('results/plus_shape-conditional_cinn_8_0.pt')['net'])
    # plot_model_conditional(c, diverse=True)

