import numpy as np
import traceback
import multiprocessing as mp
from tqdm import tqdm

from best_shape_fit import *
from data import PlusShapeModel



configs = [
        # 'plus_shape.conditional_cinn_4',
        # 'plus_shape.conditional_cinn_8',
        # 'plus_shape.conditional_hint_4_full',
        # 'plus_shape.conditional_hint_8_full',


        # 'plus_shape.unconditional_inn_4_Q',
        # 'plus_shape.unconditional_inn_8',
        # 'plus_shape.unconditional_hint_4_full',
        # 'plus_shape.unconditional_hint_8_full',

        # 'plus_shape.unconditional_inn_16',
        # 'plus_shape.unconditional_inn_32',
        # 'plus_shape.unconditional_hint_4_1',
        # 'plus_shape.unconditional_hint_8_1',

        # 'plus_shape.unconditional_hint_16_1',
        # 'plus_shape.unconditional_hint_4_2',
        # 'plus_shape.unconditional_hint_8_2',


        # 'plus_shape.unconditional_hint_4_3',


        # 'plus_shape.unconditional_hint_4_0_small',
        # 'plus_shape.unconditional_hint_8_0_small',
        # 'plus_shape.unconditional_hint_16_0_small',
        # 'plus_shape.unconditional_hint_32_0_small',

        # 'plus_shape.unconditional_hint_4_1_small',
        # 'plus_shape.unconditional_hint_8_1_small',
        # 'plus_shape.unconditional_hint_16_1_small',

        # 'plus_shape.unconditional_hint_4_2_small',
        # 'plus_shape.unconditional_hint_8_2_small',

        # 'plus_shape.unconditional_hint_4_3_small',


         # 'plus_shape.unconditional_hint_4_0_big',
        # 'plus_shape.unconditional_hint_8_0_big',
        # 'plus_shape.unconditional_hint_16_0_big',
        # 'plus_shape.unconditional_hint_32_0_big',

        # 'plus_shape.unconditional_hint_4_1_big',
        # 'plus_shape.unconditional_hint_8_1_big',
        # 'plus_shape.unconditional_hint_16_1_big',

        # 'plus_shape.unconditional_hint_4_2_big',
        # 'plus_shape.unconditional_hint_8_2_big',

        # 'plus_shape.unconditional_hint_4_3_big',
    ]


def evaluate_all():
    with mp.Pool(1) as p:
        p.map(evaluate_config, configs)


def evaluate_config(config):
    data_model = PlusShapeModel()

    try:
        results = {'IoU': [], 'DICE': [], 'max_h': [], 'avg_h': []}

        sample = np.load(f'results/{config.replace(".", "-")}_0_sample.npy')
        # print(config, sample.shape)
        # return

        curves = data_model.trace_fourier_curves(sample)
        curves_dense = data_model.trace_fourier_curves(sample, n_points=1000)

        for j in range(len(curves)):
            points = torch.tensor(curves[j]).float().cpu()
            params = fit_plus_shape_to_points(points)
            iou, dice = iou_and_dice_plus_shape(params, points)
            max_h, avg_h = max_and_avg_hausdorff_distance_plus_shape(params, curves_dense[j])
            print(config, j, iou, dice, max_h, avg_h, flush=True)

            results['IoU'].append(iou)
            results['DICE'].append(dice)
            results['max_h'].append(max_h)
            results['avg_h'].append(avg_h)

        iou   = np.mean(results['IoU'])
        dice  = np.mean(results['DICE'])
        max_h = np.mean(results['max_h'])
        avg_h = np.mean(results['avg_h'])

        print(iou, dice, max_h, avg_h)
        np.save(f'results/{config.replace(".", "-")}_0', np.stack([iou, dice, max_h, avg_h]))

    except Exception as e:
        print(f'ERROR with config "{config}"')
        print(e)
        traceback.print_exc()
        # return



def collect_results():
    for config in configs:
        results = np.array([np.load(f'results/{config.replace(".", "-")}_0.npy')])
        means, stds = results.mean(axis=0), results.std(axis=0)

        print(config)
        # print(f'iou:   {means[0]:.4f} | {stds[0]:.4f}')
        # print(f'dice:  {means[1]:.4f} | {stds[1]:.4f}')
        # print(f'max_h: {means[2]:.4f} | {stds[2]:.4f}')
        # print(f'avg_h: {means[3]:.4f} | {stds[3]:.4f}')
        print(f'{means[0]:.3f}')
        print(f'{means[3]:.3f}')
        print()



if __name__ == '__main__':
    pass

    evaluate_all()
    collect_results()
