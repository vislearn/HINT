from time import time

import torch
from FrEIA.framework import *
from FrEIA.modules import *

import monitoring


# Load training configuration, model and data set

# from configs.fourier_curve.unconditional_inn_1 import c, model, model_inverse
# from configs.fourier_curve.unconditional_inn_2 import c, model, model_inverse
# from configs.fourier_curve.unconditional_hint_1_full import c, model, model_inverse
# from configs.fourier_curve.unconditional_hint_2_full import c, model, model_inverse

# from configs.plus_shape.unconditional_inn_4 import c, model, model_inverse
# from configs.plus_shape.unconditional_inn_8 import c, model, model_inverse
# from configs.plus_shape.unconditional_inn_16 import c, model, model_inverse
# from configs.plus_shape.unconditional_inn_32 import c, model, model_inverse

# from configs.uci_data.power_hint_4 import c, model, model_inverse
# from configs.uci_data.power_hint_8 import c, model, model_inverse
# from configs.uci_data.power_inn_4 import c, model, model_inverse
# from configs.uci_data.power_inn_8 import c, model, model_inverse

# from configs.uci_data.gas_hint_4 import c, model, model_inverse
# from configs.uci_data.gas_hint_8 import c, model, model_inverse
# from configs.uci_data.gas_inn_4 import c, model, model_inverse
# from configs.uci_data.gas_inn_8 import c, model, model_inverse

# from configs.uci_data.miniboone_hint_4 import c, model, model_inverse
# from configs.uci_data.miniboone_hint_8 import c, model, model_inverse
# from configs.uci_data.miniboone_inn_4 import c, model, model_inverse
# from configs.uci_data.miniboone_inn_8 import c, model, model_inverse


def save(c, name):
    torch.save({#'opt': optim.state_dict(),
                'net': c.model.state_dict()}, name)


def load(c, name):
    state_dicts = torch.load(name)
    c.model.load_state_dict(state_dicts['net'])
    # try:
    #     optim.load_state_dict(state_dicts['opt'])
    # except ValueError:
    #     print('Cannot load optimizer for some reason or other')


def save_sample(c, N=500):
    # create sample from saved model
    try:
        load(c, f'output/{c.suffix}.pt')
    except:
        pass
    c.model.eval()
    with torch.no_grad():
        z_sample = torch.cuda.FloatTensor(N, c.ndim_z).normal_()
        y_sample = torch.tensor(y).expand(N, c.ndim_y).cuda()
        x_sample = c.model_inverse(y_sample, z_sample)
        np.save(f'output/samples/{c.suffix}_sample-{N}', x_sample.cpu())


def evaluate(c):

    with torch.no_grad():
        c.model.eval()
        loader = c.test_loader

        batch_idx = 0
        loss_history = []

        for x, y in loader:
            batch_losses = []
            batch_idx += 1
            if batch_idx > c.max_batches_per_epoch > 0: break

            x = x.to(c.device)
            x += 0.01 * torch.randn_like(x)

            # Forward pass
            z = c.model(x)
            log_jacobian = c.model.log_jacobian(x, run_forward=False)

            # Maximum likelihood loss terms
            batch_losses.append(0.5 * torch.sum(z**2, dim=1).mean())
            batch_losses.append(-log_jacobian.mean())

            # Add up all losses
            loss_total = sum(batch_losses)
            loss_history.append([l.item() for l in batch_losses])

        return np.mean(loss_history, axis=0).sum()


def train_epoch(c, optim, i_epoch, vis_latent, test=False):

    if not test:
        c.model.train()
        loader = c.train_loader

    if test:
        c.model.eval()
        loader = c.test_loader
        nograd = torch.no_grad()
        nograd.__enter__()


    batch_idx = 0
    loss_history = []

    for x, y in loader:
        optim.zero_grad()
        batch_losses = []
        batch_idx += 1
        if batch_idx > c.max_batches_per_epoch > 0: break

        x = x.to(c.device)
        x += 0.01 * torch.randn_like(x)

        # Forward pass
        z = c.model(x)
        log_jacobian = c.model.log_jacobian(x, run_forward=False)

        # Maximum likelihood loss terms
        batch_losses.append(0.5 * torch.sum(z**2, dim=1).mean())
        batch_losses.append(-log_jacobian.mean())

        # Add up all losses
        loss_total = sum(batch_losses)
        loss_history.append([l.item() for l in batch_losses])

        # Compute gradients
        if not test:
            loss_total.backward()

            # Clamp gradients
            for p in c.model.params_trainable:
                p.grad.data.clamp_(-5.00, 5.00)

            # Parameter update
            optim.step()

            # Update progress bar
            monitoring.visualizer.update_progress(batch_idx, i_epoch+1)

    if test:
        # Update plots
        latent_sample = z[:500,:].data.cpu().numpy()
        with torch.no_grad():
            vis_x = c.model_inverse(vis_latent).data.cpu().numpy()
        monitoring.visualizer.update_plots(latent_sample, vis_x)

        nograd.__exit__(None, None, None)

    return np.mean(loss_history, axis=0)


def main(c):
    loss_labels = ['-log p(z)', '-log |det(J)|']

    # Init trainable model parameters
    if c.init_scale > 0:
        for p in c.model.params_trainable:
            p.data = c.init_scale * torch.randn_like(p.data)
    # Count total number of trainable parameters
    n_model_params = sum([p.numel() for p in c.model.params_trainable])
    print(f'\nModel {c.suffix} has {n_model_params:,} trainable parameters.\n')
    # assert False

    # Prepare optimizer and learning rate schedule
    optim = torch.optim.Adam(c.model.params_trainable, lr=c.lr_init,
                             betas=c.adam_betas, eps=1e-4,
                             weight_decay=c.l2_weight_reg)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1,
                                                   gamma=(c.final_decay)**(1./c.n_epochs))

    # For visualization
    vis_batchsize = 300
    vis_latent = torch.randn(vis_batchsize, c.ndim_x).to(c.device)


    monitoring.restart(c, loss_labels)
    # monitoring.visualizer.print_config()

    t_start = time()
    try:
        for i_epoch in range(c.n_epochs):
            if i_epoch < c.pre_low_lr:
                for param_group in optim.param_groups:
                    param_group['lr'] = c.lr_init * 3e-2

            train_losses = train_epoch(c, optim, i_epoch, vis_latent)
            test_losses  = train_epoch(c, optim, i_epoch, vis_latent, test=True)

            monitoring.visualizer.update_losses(np.concatenate([train_losses, test_losses]),
                                                lr_scheduler.get_lr(), logscale=False)

            lr_scheduler.step()
        # save(c, f'output/{c.suffix}.pt')
    except:
        # save(c, f'output/{c.suffix}.pt' + '_ABORT')
        raise

    finally:
        print("\n\nTraining took %f minutes\n\n" % ((time()-t_start)/60.))

    return test_losses.sum()


if __name__ == "__main__":
    pass

    # main(c)
    # save_sample(c)
