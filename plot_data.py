import json
import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
from shapely import geometry as geo
from shapely.ops import nearest_points

from data import *



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
    # plot_dataset_example(PlusShapeModel(), limits=[-4,4,-4,4], seed=11)
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
 