import torch
import numpy as np
import traceback
from best_shape_fit import *
from tqdm import tqdm



n_eval_samples = 1000
n_runs = 1

train_configs = [
        # 'lens_shape.conditional_hint_1_full',
        # 'lens_shape.conditional_hint_2_full',
        # 'lens_shape.conditional_hint_4_full',
        # 'lens_shape.conditional_hint_8_full',
        # 'lens_shape.conditional_cinn_1',
        # 'lens_shape.conditional_cinn_2',
        # 'lens_shape.conditional_cinn_4',
        # 'lens_shape.conditional_cinn_8',


        # 'lens_shape.unconditional_hint_1_full',
        # 'lens_shape.unconditional_hint_2_full',
        # 'lens_shape.unconditional_inn_1',
        # 'lens_shape.unconditional_inn_2',


        # 'plus_shape.conditional_recursive_cinn_4',
        # 'plus_shape.unconditional_hint_4_3_reshuffle',
        # 'plus_shape.unconditional_hint_4_3_constwidth',
        # 'plus_shape.unconditional_hint_4_3_lessshrink',


        # 'plus_shape.conditional_cinn_4',
        # 'plus_shape.conditional_cinn_8',
        # 'plus_shape.conditional_hint_4_full',
        # 'plus_shape.conditional_hint_8_full',


        # 'plus_shape.unconditional_inn_4_Q',
        # 'plus_shape.unconditional_inn_4',
        # 'plus_shape.unconditional_inn_8',
        # 'plus_shape.unconditional_inn_16',
        # 'plus_shape.unconditional_inn_32',

        # 'plus_shape.unconditional_hint_4_1',
        # 'plus_shape.unconditional_hint_8_1',
        # 'plus_shape.unconditional_hint_16_1',

        # 'plus_shape.unconditional_hint_4_2',
        # 'plus_shape.unconditional_hint_8_2',

        # 'plus_shape.unconditional_hint_4_3',

        # 'plus_shape.unconditional_hint_4_full',
        # 'plus_shape.unconditional_hint_8_full',


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

eval_configs = [
        # 'plus_shape.conditional_cinn_4',
        # 'plus_shape.conditional_cinn_8',
        # 'plus_shape.conditional_hint_4_full',
        # 'plus_shape.conditional_hint_8_full',
    ]

def train_and_evaluate():
    for config in train_configs + eval_configs:
        for i in range(n_runs):
            try:
                # Import config
                exec(f'from configs.{config} import c', globals())

                # n_model_params = sum([p.numel() for p in c.model.params_trainable])
                # print(f'Model {c.suffix} has {n_model_params:,} trainable parameters.')
                # assert False

                if config in train_configs:
                    # Run appropriate training script, save weights and generate samples
                    if 'unconditional' in config:
                        from train_unconditional import main, save
                        test_loss = main(c)
                        save(c, f'results/{config.replace(".", "-")}_{i}.pt')
                        with torch.no_grad():
                            sample = c.model_inverse(torch.randn(n_eval_samples, c.data_model.n_parameters).cuda())

                    else:
                        from train_conditional import main, save
                        test_loss = main(c)
                        save(c, f'results/{config.replace(".", "-")}_{i}.pt')
                        with torch.no_grad():
                            y_target = torch.tensor(c.vis_y_target).expand(n_eval_samples, c.data_model.n_observations).cuda()
                            sample = c.model_inverse(y_target, torch.randn(n_eval_samples, c.data_model.n_parameters).cuda())

                    print(test_loss)

                elif config in eval_configs:
                    # Load pretrained weights and generate samples
                    state_dicts = torch.load(f'output/{c.suffix}.pt')
                    # state_dicts = torch.load(f'results/{config.replace(".", "-")}_{i}.pt')
                    c.model.load_state_dict(state_dicts['net'])

                    with torch.no_grad():
                        if 'unconditional' in config:
                            sample = c.model_inverse(torch.randn(n_eval_samples, c.data_model.n_parameters).cuda())
                        else:
                            y_target = torch.tensor(c.vis_y_target).expand(n_eval_samples, c.data_model.n_observations).cuda()
                            sample = c.model_inverse(y_target, torch.randn(n_eval_samples, c.data_model.n_parameters).cuda())

                # Evaluate model and save everything
                sample = c.data_model.unflatten_coeffs(sample.cpu().numpy())
                np.save(f'results/{config.replace(".", "-")}_{i}_sample', sample)

                if 'lens' in config:
                    device = 'cpu'
                    results = {'IoU': [], 'DICE': [], 'max_h': [], 'avg_h': []}
                    curves = c.data_model.trace_fourier_curves(sample)
                    curves_dense = c.data_model.trace_fourier_curves(sample, n_points=1000)

                    for j in tqdm(range(len(curves))):
                        points = torch.tensor(curves[j]).float().to(device)
                        params = fit_lens_shape_to_points(points)
                        iou, dice = iou_and_dice_lens_shape(params, points)
                        max_h, avg_h = max_and_avg_hausdorff_distance_lens_shape(params, curves_dense[j])

                        results['IoU'].append(iou)
                        results['DICE'].append(dice)
                        results['max_h'].append(max_h)
                        results['avg_h'].append(avg_h)

                    iou   = np.mean(results['IoU'])
                    dice  = np.mean(results['DICE'])
                    max_h = np.mean(results['max_h'])
                    avg_h = np.mean(results['avg_h'])

                    print(iou, dice, max_h, avg_h)
                    np.save(f'results/{config.replace(".", "-")}_{i}', np.stack([iou, dice, max_h, avg_h]))

            except Exception as e:
                print(f'ERROR with config "{config}"', i)
                print(e)
                traceback.print_exc()
                # return



def collect_results():
    for config in train_configs + eval_configs:
        results = np.array([np.load(f'results/{config.replace(".", "-")}_{i}.npy') for i in range(n_runs)])
        means, stds = results.mean(axis=0), results.std(axis=0)

        print(config)
        # print(f'iou:   {means[0]:.3f} \\pm {stds[0]:.3f}')
        # print(f'dice:  {means[1]:.3f} \\pm {stds[1]:.3f}')
        # print(f'max_h: {means[2]:.3f} \\pm {stds[2]:.3f}')
        # print(f'avg_h: {means[3]:.3f} \\pm {stds[3]:.3f}')
        print(f'{means[0]:.3f} \\pm {stds[0]:.3f} & {means[3]:.3f} \\pm {stds[3]:.3f}')
        print()



def test_likelihood():
    for config in train_configs + eval_configs:
        try:
            # Import config
            exec(f'from configs.{config} import c', globals())
            if 'unconditional' in config:
                from train_unconditional import evaluate
            else:
                from train_conditional import evaluate

            likelihoods = []
            corr_mses = []
            for i in range(n_runs):
                # Load model
                state_dicts = torch.load(f'results/{config.replace(".", "-")}_{i}.pt')
                state = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
                c.model.load_state_dict(state, strict=False)
                # Evaluate likelihood
                likelihoods.append(-evaluate(c) / c.data_model.n_parameters)
                # Generate sample and compare parameter correlations
                with torch.no_grad():
                    if 'unconditional' in config:
                        x = c.model_inverse(torch.randn(10000, c.ndim_z).to(c.device)).detach().cpu().numpy()
                        corr = np.corrcoef(x.T)
                        corr_true = np.load(f'data/{c.data_model.name}_corr.npy')
                    else:
                        x = c.model_inverse(torch.randn(4000, c.ndim_z).to(c.device)).detach().cpu().numpy()
                        corr = np.corrcoef(x.T)
                        corr_true = np.load(f'data/{c.data_model.name}_corr_conditional.npy')
                corr_mses.append(np.nanmean(np.square(corr - corr_true)))

            print(config)
            if n_runs > 1:
                print(f'{np.mean(likelihoods):.3f} \pm {np.std(likelihoods):.3f}')
                print(f'{np.mean(corr_mses):.4f} \pm {np.std(corr_mses):.4f}')
            else:
                print(f'{likelihoods[0]:.3f}')
                print(f'{corr_mses[0]:.4f}')
            print()

        except Exception as e:
            print(f'ERROR with config "{config}"', i)
            print(e)
            # traceback.print_exc()
            # return



if __name__ == '__main__':
    pass

    # train_and_evaluate()
    # collect_results()
    test_likelihood()
