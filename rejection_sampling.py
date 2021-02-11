import numpy as np
import torch
import time
import os
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from tqdm import tqdm
from scipy.spatial import distance_matrix

from data import *


# Import and organize all the models
# dataset = ('fourier_curve', 'fourier-curve')
dataset = ('lens_shape', 'lens-shape')
# dataset = ('plus_shape', 'plus-shape')
run = 0
conditional_models = {}
# for name in ['conditional_cinn_1', 'conditional_cinn_2', 'conditional_cinn_4', 'conditional_cinn_8', 'conditional_hint_1_full', 'conditional_hint_2_full', 'conditional_hint_4_full', 'conditional_hint_8_full']:
#     exec("import configs." + dataset[0] + "." + name + " as " + name)
#     # exec(name + ".model.load_state_dict(torch.load(f'output/{run}/{" + name + ".c.suffix}.pt')['net'])")
#     exec(name + ".model.load_state_dict(torch.load(f'results/{dataset[0]}-{name}_{run}.pt')['net'])")
#     exec("conditional_models[" + name + ".c.suffix] = {'model': " + name + ".model, 'inverse': " + name + ".model_inverse}")
unconditional_models = {}
# for name in ['unconditional_inn_1', 'unconditional_inn_2', 'unconditional_hint_1_full', 'unconditional_hint_2_full']:
#     exec("import configs." + dataset[0] + "." + name + " as " + name)
#     # exec(name + ".model.load_state_dict(torch.load(f'output/{run}/{" + name + ".c.suffix}.pt')['net'])")
#     exec(name + ".model.load_state_dict(torch.load(f'results/{dataset[0]}-{name}_{run}.pt')['net'])")
#     exec("unconditional_models[" + name + ".c.suffix] = {'model': " + name + ".model, 'inverse': " + name + ".model_inverse}")


def check_mmd_kernels(x, y, widths_exponents=[(1, 0.5), (0.2, 0.8), (0.2, 0.4)]):
    # compute distance matrices dxx, dyy, dxy
    xx, yy, xy = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    dxx = torch.clamp(rx.t() + rx - 2.*xx, 0, np.inf)
    dyy = torch.clamp(ry.t() + ry - 2.*yy, 0, np.inf)
    dxy = torch.clamp(rx.t() + ry - 2.*xy, 0, np.inf)
    # plot histogram
    bins = np.linspace(0, 20, 100)
    plt.hist(dxx.data.cpu().numpy().reshape(-1), bins, alpha=.5, label='dxx', density=True)
    plt.hist(dyy.data.cpu().numpy().reshape(-1), bins, alpha=.5, label='dyy', density=True)
    plt.hist(dxy.data.cpu().numpy().reshape(-1), bins, alpha=.5, label='dxy', density=True)
    # overlay kernels
    multi_kernel = np.zeros(*bins.shape)
    for C, a in widths_exponents:
        multi_kernel += C**a * ((C+bins)/a)**-a
        plt.plot(bins, C**a * ((C+bins)/a)**-a, label=f'{C:.2f}/{a:.2f}')
    plt.plot(bins, multi_kernel, lw=2, c='k', label='sum')
    plt.legend(loc='upper right')
    plt.show()


def multi_mmd(x, y, widths_exponents=[(0.5, 1), (0.2, 1), (0.2, 0.5)]):
# def multi_mmd(x, y, widths_exponents=[(1, 0.5), (0.2, 0.8), (0.2, 0.4)]):
    xx, yy, xy = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    dxx = torch.clamp(rx.t() + rx - 2.*xx, 0, np.inf)
    dyy = torch.clamp(ry.t() + ry - 2.*yy, 0, np.inf)
    dxy = torch.clamp(rx.t() + ry - 2.*xy, 0, np.inf)

    XX = torch.zeros(xx.shape).cuda()
    YY = torch.zeros(xx.shape).cuda()
    XY = torch.zeros(xx.shape).cuda()
    for C, a in widths_exponents:
        XX += C**a * ((C + dxx) / a)**-a
        YY += C**a * ((C + dyy) / a)**-a
        XY += C**a * ((C + dxy) / a)**-a

    return torch.mean(XX + YY - 2.*XY)


def prepare_samples(model, N=int(1e8)):
    print(f'Drawing {N:,} samples from "{model.name}" prior...', end=' ')
    t = time.time()
    x, y, = [], []
    for i in tqdm(range(int(N/1e4))):
        x.append(model.sample_prior(int(1e4)).astype(np.float32))
        y.append(model.forward_process(x[-1]).astype(np.float32))
    np.save(f'abc/{model.name}_x_huge', np.concatenate(x, axis=0))
    np.save(f'abc/{model.name}_y_huge', np.concatenate(y, axis=0))
    print(f'Done in {time.time()-t:.1f} seconds.')


def quantile_ABC(x, y, y_target, n=4000):
    print(f'Evaluating ABC to obtain {n:,} samples closest to {y_target[0]} from set of {len(y):,}...', end=' ')
    t = time.time()
    d = distance_matrix(y_target, y)[0]
    sort = np.argsort(d)[1:]
    sample = x[sort][:n]
    threshold = d[sort[n]]
    print(f'Done in {time.time()-t:.1f} seconds, tolerance is {threshold:.3f}.')
    return sample, threshold


def mean_target_distance(model, y_target, x):
    y = model.forward_process(x.cpu().numpy())#[:,2:]
    dist = torch.sum((torch.FloatTensor(y) - y_target[0].cpu())**2, dim=1).sqrt()
    return dist.mean()


def correlation_conditional(data_model, y_target, n=4000):
    y_target = np.array(y_target)
    try:
        sample = np.load(f'data/{data_model.name}_corr_conditional_sample.npy')
    except:
        if data_model.name == 'lens-shape':
            x, y = np.load(f'abc/{data_model.name}_x_huge.npy'), np.load(f'abc/{data_model.name}_y_huge.npy')
            sample, _ = quantile_ABC(x, y, [y_target], n=n)
        if data_model.name == 'plus-shape':
            samples = []
            labels = []
            while len(samples) < n:
                coords, label = data_model.generate_plus_shape(forward=True, target=y_target)
                d = np.sqrt(np.sum(np.square(y_target - label)))
                if d < 0.05:
                    sample = data_model.fourier_coeffs(coords, n_coeffs=PlusShapeModel.n_parameters//4)
                    samples.append(sample)
                    labels.append(label)
                    print(f' {len(samples)} ', end='', flush=True)
                print('.', end='')
            sample = np.stack(samples)
            labels = np.stack(labels)
            sample = data_model.flatten_coeffs(sample)

    np.save(f'data/{data_model.name}_corr_conditional_sample.npy', sample)
    corr = np.corrcoef(sample.T)
    np.save(f'data/{data_model.name}_corr_conditional.npy', corr)
    print(corr.shape)


def compare_unconditional(data_model, n_runs=100, sample_size=4000):
    # Load data
    x = np.load(f'abc/{data_model.name}_x_huge.npy')
    # Prepare lists
    for model in unconditional_models.values():
        model['samples'] = []
        model['times'] = []
        model['mmds'] = []
    # Perform runs
    for i in range(n_runs):
        print(f'\nRun {i+1:04}/{n_runs:04}:')
        # Ground truth sample and shared latent sample for all models
        gt_sample = torch.tensor(x[np.random.choice(x.shape[0], sample_size, replace=False)], device='cuda')
        z_sample = torch.cuda.FloatTensor(sample_size, data_model.n_parameters).normal_()
        # Generate samples from all models
        with torch.no_grad():
            for name, model in unconditional_models.items():
                t = time.time()
                sample = model['inverse'](z_sample)
                model['times'].append(time.time() - t)
                model['samples'].append(sample)
                model['mmds'].append(multi_mmd(sample, gt_sample).item())
                print(f"{name+':':48} {model['mmds'][-1]:.5f}     ({model['times'][-1]:.3f}s)")
    # Print averaged results
    print('\nAverage over all runs:')
    for name, model in unconditional_models.items():
        print(f"{name+':':45} {np.mean(model['mmds']):.5f}     ({np.mean(model['times']):.3f}s)")
    # Save results for later plotting
    dump = {name: {'times': model['times'], 'mmds': model['mmds']} for (name, model) in unconditional_models.items()}
    with open(f'abc/{data_model.name}_unconditional_comparison_{run}.pkl', 'wb') as f:
        pickle.dump(dump, f)


def compare_conditional(data_model, n_runs=1000, sample_size=4000):
    # Load data
    x, y = np.load(f'abc/{data_model.name}_x_huge.npy'), np.load(f'abc/{data_model.name}_y_huge.npy')
    # Prepare lists
    for model in conditional_models.values():
        model['samples'] = []
        model['times'] = []
        model['mmds'] = []
        model['dists'] = []
    # Perform runs
    for i in range(n_runs):
        # Rejection sample to compare against
        try:
            with open(f'abc/{data_model.name}/{i:05}.pkl', 'rb') as f:
                y_target, gt_sample, threshold = pickle.load(f)
            assert gt_sample.shape[0] >= sample_size
        except:
            if not os.path.exists(f'abc/{data_model.name}'):
                os.mkdir(f'abc/{data_model.name}')
            y_target = data_model.forward_process(data_model.sample_prior(1)).astype(np.float32)
            gt_sample, threshold = quantile_ABC(x, y, y_target, n=sample_size)
            with open(f'abc/{data_model.name}/{i:05}.pkl', 'wb') as f:
                pickle.dump((y_target, gt_sample, threshold), f)
        print(f'\nRun {i+1:04}/{n_runs:04} | y = {np.round(y_target[0], 3)}:')
        gt_sample = torch.from_numpy(gt_sample).cuda()
        # Shared latent sample and target observation for all models
        z_sample = torch.cuda.FloatTensor(sample_size, data_model.n_parameters).normal_()
        y_target = torch.tensor(y_target).expand(sample_size, data_model.n_observations).cuda()
        # Generate samples from all models
        with torch.no_grad():
            for name, model in conditional_models.items():
                t = time.time()
                sample = model['inverse'](y_target, z_sample)
                model['times'].append(time.time() - t)
                model['samples'].append(sample)
                model['mmds'].append(multi_mmd(sample, gt_sample).item())
                model['dists'].append(mean_target_distance(data_model, y_target, sample).item())
                print(f"{name+':':46} {model['mmds'][-1]:.5f}     {model['dists'][-1]:.5f}     ({model['times'][-1]:.3f}s)")
    # Print averaged results
    print('\nAverage over all runs:')
    for name, model in conditional_models.items():
        print(f"{name+':':45} {np.mean(model['mmds']):.5f}     {np.mean(model['dists']):.5f}     ({np.mean(model['times']):.3f}s)")
    # Save results for later plotting
    dump = {name: {'times': model['times'], 'mmds': model['mmds'], 'dists': model['dists']} for (name, model) in conditional_models.items()}
    with open(f'abc/{data_model.name}_conditional_comparison_{run}.pkl', 'wb') as f:
        pickle.dump(dump, f)


def accumulate_metrics_unconditional():
    mmds = {'lens-shape_unconditional_inn-1':[], 'lens-shape_unconditional_inn-2':[], 'lens-shape_unconditional_hint-1-full':[], 'lens-shape_unconditional_hint-2-full':[]}
    for i in range(3):
        with open(f'abc/lens-shape_unconditional_comparison_{i}.pkl', 'rb') as f:
            d = pickle.load(f)
        for name, model in d.items():
            # print(name, np.mean(model['mmds']), np.std(model['mmds']))
            mmds[name].append(np.mean(model['mmds']))
    for name in mmds.keys():
        print(name)
        print(f'{np.nanmean(mmds[name]):.3f} \\pm {np.nanstd(mmds[name]):.3f}')
        print()


def accumulate_metrics_conditional():
    # mmds = {'fourier-curve_conditional_cinn-1':[], 'fourier-curve_conditional_cinn-2':[], 'fourier-curve_conditional_cinn-4':[], 'fourier-curve_conditional_cinn-8':[], 'fourier-curve_conditional_hint-1-full':[], 'fourier-curve_conditional_hint-2-full':[], 'fourier-curve_conditional_hint-4-full':[], 'fourier-curve_conditional_hint-8-full':[]}
    mmds = {'lens-shape_conditional_cinn-1':[], 'lens-shape_conditional_cinn-2':[], 'lens-shape_conditional_cinn-4':[], 'lens-shape_conditional_cinn-8':[], 'lens-shape_conditional_hint-1-full':[], 'lens-shape_conditional_hint-2-full':[], 'lens-shape_conditional_hint-4-full':[], 'lens-shape_conditional_hint-8-full':[]}
    for i in range(3):
        # with open(f'abc/fourier-curve_conditional_comparison_{i+1}.pkl', 'rb') as f:
        with open(f'abc/lens-shape_conditional_comparison_{i}.pkl', 'rb') as f:
            d = pickle.load(f)
        for name, model in d.items():
            # print(name, np.mean(model['mmds']), np.std(model['mmds']))
            mmds[name].append(np.mean(model['mmds']))
    for name in mmds.keys():
        print(name)
        print(f'{np.nanmean(mmds[name]):.3f} \\pm {np.nanstd(mmds[name]):.3f}')
        print()



if __name__ == '__main__':
    pass

    # prepare_samples(FourierCurveModel())
    # prepare_samples(LensShapeModel())

    # correlation_conditional(PlusShapeModel(), (0.75, 0.0, 1.0, 3.0), n=4000)
    # correlation_conditional(LensShapeModel(), (2.0, -1.0), n=4000)

    # x = torch.tensor(np.load('abc/fourier-curve_x_huge.npy')[:4000]).cuda()
    # y = torch.tensor(np.load('abc/fourier-curve_x_huge.npy')[-4000:]).cuda()
    # check_mmd_kernels(x,y)

    # compare_conditional(FourierCurveModel())
    # compare_unconditional(FourierCurveModel())

    # compare_conditional(LensShapeModel())
    # compare_unconditional(LensShapeModel())

    # accumulate_metrics_unconditional()
    # accumulate_metrics_conditional()
