import torch
import torch.nn.functional as F
from shapely import geometry as geo
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
from time import time

from data import *

device = 'cpu'
# device = 'cuda' # weird numerical results for some reason



def squared_dists_points_to_line_segment(points, a, b):
    n = b - a
    dist = (n*n).sum().sqrt()
    n = n/dist
    ap = a[None,:] - points
    length = torch.max(a.new_zeros(1), torch.min(dist, -torch.matmul(ap, n)))
    shortest_vector_to_segment = ap + length[:,None] * n[None,:]
    return (shortest_vector_to_segment**2).sum(dim=1)



def plus_segments_from_params(params):
    xlength, ylength, xwidth, ywidth, xshift, yshift, xoffset, yoffset, angle = params
    xleft, xbottom, xright, xtop = xshift - xlength/2, -xwidth/2, xshift + xlength/2, xwidth/2
    yleft, ybottom, yright, ytop = -ywidth/2, yshift - ylength/2, ywidth/2, yshift + ylength/2
    xleft = torch.min(xleft, yleft - 0.01)
    xright = torch.max(xright, yright + 0.01)
    ytop = torch.max(ytop, xtop + 0.01)
    ybottom = torch.min(ybottom, xbottom - 0.01)
    segments = [torch.stack([torch.cat([xleft, xtop]), torch.cat([yleft, xtop])]),
                torch.stack([torch.cat([yleft, xtop]), torch.cat([yleft, ytop])]),
                torch.stack([torch.cat([yleft, ytop]), torch.cat([yright, ytop])]),
                torch.stack([torch.cat([yright, ytop]), torch.cat([yright, xtop])]),
                torch.stack([torch.cat([yright, xtop]), torch.cat([xright, xtop])]),
                torch.stack([torch.cat([xright, xtop]), torch.cat([xright, xbottom])]),
                torch.stack([torch.cat([xright, xbottom]), torch.cat([yright, xbottom])]),
                torch.stack([torch.cat([yright, xbottom]), torch.cat([yright, ybottom])]),
                torch.stack([torch.cat([yright, ybottom]), torch.cat([yleft, ybottom])]),
                torch.stack([torch.cat([yleft, ybottom]), torch.cat([yleft, xbottom])]),
                torch.stack([torch.cat([yleft, xbottom]), torch.cat([xleft, xbottom])]),
                torch.stack([torch.cat([xleft, xbottom]), torch.cat([xleft, xtop])])]
    segments = torch.stack([s for s in segments if ((s[0]-s[1])**2).sum() > 0])
    R = torch.stack([torch.cat((torch.cos(angle), torch.sin(angle))), torch.cat((-torch.sin(angle), torch.cos(angle)))])
    segments = torch.matmul(segments, R)
    segments = segments + torch.cat([xoffset, yoffset])
    return segments



def points_to_plus_loss(points, params, corner_weight=1):
    # from shape parameters, extract all line segments
    segments = plus_segments_from_params(params)
    # for each point in generated shape, compute distance to each segment
    dists = []
    for segment in segments:
        dists.append(squared_dists_points_to_line_segment(points, segment[0], segment[1]))
    dists = torch.stack(dists)
    # for each point, take minimum distance over all segments
    dists = torch.min(dists, dim=0)[0]
    corner_dists = torch.min(torch.cdist(segments[:,0,:].contiguous(), points), dim=-1)[0]**2
    return dists.mean() + corner_weight * corner_dists.mean()



def init_plus_params(angle, xshift=0, yshift=0, center=np.array([0,0])):
    xlength = torch.nn.Parameter(torch.ones(1, device=device) * 5)
    ylength = torch.nn.Parameter(torch.ones(1, device=device) * 5)
    xwidth = torch.nn.Parameter(torch.ones(1, device=device) * 2)
    ywidth = torch.nn.Parameter(torch.ones(1, device=device) * 2)
    xshift = torch.nn.Parameter(torch.ones(1, device=device) * xshift)
    yshift = torch.nn.Parameter(torch.ones(1, device=device) * yshift)
    xoffset = torch.nn.Parameter(torch.ones(1, device=device) * center[0])
    yoffset = torch.nn.Parameter(torch.ones(1, device=device) * center[1])
    angle = torch.nn.Parameter(torch.ones(1, device=device) * angle)
    return [xlength, ylength, xwidth, ywidth, xshift, yshift, xoffset, yoffset, angle]



def fit_line(points):
    from sklearn import linear_model
    ransac = linear_model.RANSACRegressor(residual_threshold=0.05)
    ransac.fit(points[:,0,None], points[:,1,None])
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    return np.concatenate([np.array([[0], [1]]), ransac.predict([[0], [1]])], axis=1)



def fit_plus_shape_to_points(points, verbose=False):
    # Get dominant angle by fitting a line with RANSAC
    line = fit_line(points.cpu())
    line -= line[0,:]
    angle = np.arctan2(line[1,1], 1)

    # Combinations of x and y shifts to try out
    xyshifts = [(0, 0), (-1.5, -1.5), (-1.5, 0), (-1.5, 1.5), (0, -1.5), (0, 1.5), (1.5, -1.5), (1.5, 0), (1.5, 1.5)]

    results = []
    for xyshift in xyshifts:
        # Init Plus shape parameters with dominant angle and center of the shape
        params = init_plus_params(angle, xshift=xyshift[0], yshift=xyshift[1], center=points.mean(axis=0))

        # Init optimizer and learning rate scheduler, much lower LR for angle
        optim = torch.optim.SGD([{'params': params[:-1]},
                                 {'params': params[-1:], 'lr': 0.01}], lr=0.1, momentum=0.2)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=(0.1)**(1./400))

        # Optimize parameters to fit Plus shape
        for i in range(400):
            optim.zero_grad()
            loss = points_to_plus_loss(points, params, corner_weight=1-i/400)
            loss.backward()
            optim.step()
            lr_scheduler.step()
            if(i==0 or i%50==49) and verbose:
                print(f'{i:03d} | {loss.item():.5f}')

        results.append((loss.item(), [p.detach() for p in params]))
        if loss.item() < 0.005:
            break
        if verbose:
            print()

    # Return parameters with the best fit
    return sorted(results)[0][1]



def iou_and_dice_plus_shape(params, points):
    segments = plus_segments_from_params(params)
    plus = geo.Polygon(segments.cpu().numpy()[:,0,:]).buffer(0)
    curve = geo.Polygon(points).buffer(0)
    intersection = plus.intersection(curve)
    union = plus.union(curve)
    return intersection.area / union.area, 2*intersection.area / (plus.area + curve.area)



def max_and_avg_hausdorff_distance(path_a, path_b):
    # Compute pairwise distances
    diffs = path_a[None,:,:] - path_b[:,None,:]
    dists = np.sqrt(np.sum(diffs*diffs, axis=-1))
    # Take minimum in each row and column and average
    minima = np.append(np.amin(dists, axis=0), np.amin(dists, axis=1))
    return np.amax(minima), np.mean(minima)



def max_and_avg_hausdorff_distance_plus_shape(params, points):
    segments = plus_segments_from_params(params)
    plus = PlusShapeModel().densify_polyline(segments.cpu().numpy()[:,0,:], max_dist=0.02)
    return max_and_avg_hausdorff_distance(plus, points)



def check_plus_shape_fitting():
    # Create example Fourier Plus shape
    model = PlusShapeModel()
    true_plus = model.generate_plus_shape()
    true_plus_fourier = model.fourier_coeffs(true_plus, n_coeffs=PlusShapeModel.n_parameters//4)
    plus_curve = model.trace_fourier_curves(true_plus_fourier[None,...])[0]
    plus_curve_dense = model.trace_fourier_curves(true_plus_fourier[None,...], n_points=1000)[0]
    points = torch.tensor(plus_curve).float().to(device)

    # # Distort a bit
    # points[:,1] += 0.4 * points[:,0]
    # plus_curve_dense[:,1] += 0.4 * plus_curve_dense[:,0]

    # Fit shape to points
    start = time()
    params = fit_plus_shape_to_points(points, verbose=True)
    iou, dice = iou_and_dice_plus_shape(params, points)
    print(f'\nIoU:  {iou:.3f}')
    print(f'DICE: {dice:.3f}')
    max_h, avg_h = max_and_avg_hausdorff_distance_plus_shape(params, plus_curve_dense)
    print(f'max Hausdorff: {max_h:.3f}')
    print(f'avg Hausdorff: {avg_h:.3f}')
    print(f'\nOptimization took {time() - start:.2f} seconds\n')

    # Plot result
    segments = plus_segments_from_params(params).cpu().numpy()
    plt.figure()
    for segment in segments:
        plt.plot(segment[:,0], segment[:,1], c=(0,0,0))
    plt.plot(plus_curve_dense[:,0], plus_curve_dense[:,1], c=(1,0,0))
    plt.axis('equal')
    plt.show()



def lens_points_from_params(prototype, params):
    # Apply parameters
    x, y, scale, angle = params
    R = torch.stack([torch.cat((torch.cos(angle), torch.sin(angle))), torch.cat((-torch.sin(angle), torch.cos(angle)))])
    return torch.matmul(prototype, R) * scale + torch.cat([x,y])[None,:]



def points_to_lens_loss(prototype, points, params, lens_fit_weight=1):
    # Get points of parameterized lens
    lens = lens_points_from_params(prototype, params)
    # Compute squared distance to each point in generated shape
    dists = ((lens[None,:,:] - points[:,None,:])**2).sum(dim=-1)
    # Return mean of minima in both directions
    return dists.min(dim=1)[0].mean() + lens_fit_weight * dists.min(dim=0)[0].mean()



def init_lens_params(angle=0, scale=2, center=np.array([0,0])):
    x = torch.nn.Parameter(torch.ones(1, device=device) * center[0])
    y = torch.nn.Parameter(torch.ones(1, device=device) * center[1])
    scale = torch.nn.Parameter(torch.ones(1, device=device) * scale)
    angle = torch.nn.Parameter(torch.ones(1, device=device) * angle)
    return [x, y, scale, angle]



def get_lens_prototype():
    p0 = geo.Point(0.0, 0.0).buffer(1.5, resolution=64)
    p1 = geo.Point(3.6, 0.0).buffer(3.0, resolution=64)
    prototype = np.array(p0.intersection(p1).exterior.coords)
    return torch.tensor(prototype - prototype.mean(axis=0)).float().to(device)



def fit_lens_shape_to_points(points, verbose=False):
    # Get dominant angle by finding most distant points
    d = squareform(pdist(points))
    max_idx = np.unravel_index(d.argmax(), d.shape)
    p0, p1 = points[max_idx[0]], points[max_idx[1]]
    angle = -np.arctan2((p1-p0)[0], (p1-p0)[1])

    results = []
    for angle in [angle, (angle + np.pi)%(2*np.pi)]:
        # Init Lens shape parameters with dominant angle and center of the shape
        params = init_lens_params(angle=angle, scale=2, center=points.mean(axis=0))
        prototype = get_lens_prototype()

        # Init optimizer and learning rate scheduler, much lower LR for angle
        optim = torch.optim.SGD([{'params': params[:-1]},
                                 {'params': params[-1:], 'lr': 0.01}], lr=0.1, momentum=0.2)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=(0.1)**(1./100))

        # Optimize parameters to fit Lens shape
        for i in range(100):
            optim.zero_grad()
            loss = points_to_lens_loss(prototype, points, params)
            loss.backward()
            optim.step()
            lr_scheduler.step()
            if(i==0 or i%20==19) and verbose:
                print(f'{i:03d} | {loss.item():.5f}')

        results.append((loss.item(), [p.detach() for p in params]))

    # Return parameters with the best fit
    return sorted(results)[0][1]



def iou_and_dice_lens_shape(params, points):
    lens = lens_points_from_params(get_lens_prototype(), params).detach().cpu().numpy()
    lens = geo.Polygon(lens).buffer(0)
    curve = geo.Polygon(points).buffer(0)
    intersection = lens.intersection(curve)
    union = lens.union(curve)
    return intersection.area / union.area, 2*intersection.area / (lens.area + curve.area)



def max_and_avg_hausdorff_distance_lens_shape(params, points):
    lens = lens_points_from_params(get_lens_prototype(), params).detach().cpu().numpy()
    return max_and_avg_hausdorff_distance(lens, points)



def check_lens_shape_fitting():
    # Create example Fourier Plus shape
    model = LensShapeModel()
    true_lens = model.generate_lens_shape()
    true_lens_fourier = model.fourier_coeffs(true_lens, n_coeffs=PlusShapeModel.n_parameters//4)
    lens_curve = model.trace_fourier_curves(true_lens_fourier[None,...])[0]
    lens_curve_dense = model.trace_fourier_curves(true_lens_fourier[None,...], n_points=1000)[0]
    points = torch.tensor(lens_curve).float().to(device)

    # # Distort a bit
    # points[:,1] += 0.4 * points[:,0]
    # lens_curve_dense[:,1] += 0.4 * lens_curve_dense[:,0]

    # Fit shape to points
    start = time()
    params = fit_lens_shape_to_points(points, verbose=True)
    iou, dice = iou_and_dice_lens_shape(params, points)
    print(f'\nIoU:  {iou:.3f}')
    print(f'DICE: {dice:.3f}')
    max_h, avg_h = max_and_avg_hausdorff_distance_lens_shape(params, lens_curve_dense)
    print(f'max Hausdorff: {max_h:.3f}')
    print(f'avg Hausdorff: {avg_h:.3f}')
    print(f'\nOptimization took {time() - start:.2f} seconds\n')

    # Plot result
    plt.figure()
    plt.axvline(0, linestyle=':', c=(.5,.5,.5))
    plt.axhline(0, linestyle=':', c=(.5,.5,.5))
    lens = lens_points_from_params(get_lens_prototype(), params).detach().cpu().numpy()
    plt.plot(lens[:,0], lens[:,1], c=(0,0,0))
    plt.plot(lens_curve_dense[:,0], lens_curve_dense[:,1], c=(1,0,0))
    plt.axis('equal')
    plt.show()



if __name__ == '__main__':
    # check_plus_shape_fitting()
    check_lens_shape_fitting()
