import torch
import numpy as np
import traceback
from best_shape_fit import *
from tqdm import tqdm



n_runs = 3

configs = [
        'uci_data.power_hint_4',
        'uci_data.power_hint_8',
        'uci_data.power_inn_4',
        'uci_data.power_inn_8',

        'uci_data.gas_hint_4',
        'uci_data.gas_hint_8',
        'uci_data.gas_inn_4',
        'uci_data.gas_inn_8',

        'uci_data.miniboone_hint_4',
        'uci_data.miniboone_hint_8',
        'uci_data.miniboone_inn_4',
        'uci_data.miniboone_inn_8',
    ]


def train_and_eval():
    for config in configs:
        test_losses = []
        for i in range(n_runs):
            try:
                # Import config
                exec(f'from configs.{config} import c', globals())

                # n_model_params = sum([p.numel() for p in c.model.params_trainable])
                # print(f'Model {c.suffix} has {n_model_params:,} trainable parameters.')

                from train_unconditional import main, save, load, evaluate
                test_losses.append(main(c))
                save(c, f'results/{config.replace(".", "-")}_{i}.pt')
                # load(c, f'results/{config.replace(".", "-")}_{i}.pt')
                # test_losses.append(evaluate(c))

            except Exception as e:
                print(f'ERROR with config "{config}"', i)
                # print(e)
                # traceback.print_exc()

        print(config)
        print(test_losses)
        np.save(f'results/{config.replace(".", "-")}', np.array(test_losses))



def collect_results():
    # from data import Power, Gas, Miniboone

    for config in configs:
        if 'power' in config:
            n_dims = Power.n_parameters
            # mean, std = Power.mean_and_std()
        if 'gas' in config:
            n_dims = Gas.n_parameters
            # mean, std = Gas.mean_and_std()
        if 'miniboone' in config:
            n_dims = Miniboone.n_parameters
            # mean, std = Miniboone.mean_and_std()

        test_losses = -np.load(f'results/{config.replace(".", "-")}.npy')
        test_losses -= np.log(2*np.pi) * (n_dims/2)

        print(config)
        print(f'{test_losses.mean():.3f} \pm {test_losses.std():.3f}')
        # print('mean:', test_losses.mean())
        # print('std: ', test_losses.std())
        print()



if __name__ == '__main__':
    pass

    # train_and_eval()
    collect_results()

    # print(Power.mean_and_std(), '\n')
    # print(Gas.mean_and_std(), '\n')
    # print(Miniboone.mean_and_std(), '\n')
