import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import visdom


class Visualizer:

    def __init__(self, c, loss_labels):
        self.c = c
        self.loss_labels = loss_labels
        self.loss_labels += self.loss_labels
        self.n_losses = len(self.loss_labels)
        self.epoch = 0
        self.col_width = max(13, *[len(l)+2 for l in self.loss_labels])

    def update_losses(self, losses):
        # Table header
        if self.epoch == 0:
            print('\n Epoch |', end='')
            for i, l in enumerate(self.loss_labels):
                if i == self.n_losses//2:
                    print('  |', end='')
                print(f'{l:>{self.col_width}s}', end='')
            print('\n' + '-'*(8 + self.n_losses*self.col_width + 3))
        # Current row
        print(f'\r{self.epoch:>5d}  |', end='')
        for i, l in enumerate(losses):
            if i == self.n_losses//2:
                print('  |', end='')
            print(f'{l: {self.col_width}.4e}', end='')
        # Bits per dimension on test set
        bpd = -np.sum(losses[self.n_losses//2:]) / (self.c.ndim_z * np.log(2.))
        print(f'{bpd: {self.col_width}.4f}')

        self.epoch += 1

    def print_config(self):
        # print('='*80 + '\n')
        print('Training configuration:')
        for v in dir(self.c):
            if (v[0] == '_') or (v in ['count', 'index']):
                continue
            s = eval(f'self.c.{v}')
            print(f'    {v:25}\t{s}')
        # print('='*80 + '\n')
        print()


class LiveVisualizer(Visualizer):

    def __init__(self, c, loss_labels):
        super().__init__(c, loss_labels)
        self.visdom_handle = visdom.Visdom(env=c.suffix)
        self.visdom_handle.close()

        # Loss trajectories
        cmap = plt.get_cmap('tab10')
        traces = [dict(x=[0], y=[0], mode='lines',
                       line={'dash':'dot' if (i+1)>self.n_losses/2 else 'solid', 'width':2, 'color':colors.rgb2hex(cmap(i % (self.n_losses//2)))},
                       type='custom', name=self.loss_labels[i]) for i in range(self.n_losses)]
        layout = {'title': f'LR = {self.c.lr_init:.2e}', 'width': 800, 'height': 400, 'margin': {'l': 40, 'r': 0, 'b': 20, 't': 30, 'pad': 0},
                  'ytickmin': -10, 'ytickmax': 10, 'ytickstep': 1}
        self.visdom_handle._send({'data': traces, 'layout': layout, 'win': 'loss_trajectories'})

        # Application-specific plot
        if callable(getattr(self.c.data_model, "init_plot", None)):
            self.application_plot = self.visdom_handle.matplot(self.c.data_model.init_plot(self.c.vis_y_target))

        # Plot for samples from marginal latent distribution
        self.latent_plot = self.visdom_handle.scatter(X=np.zeros((1,2)), opts={'margin': {'l': 0, 'r': 0, 'b': 0, 't': 0, 'pad': 0}})

        # Pie charts for progress
        self.update_progress(1, 1)

    def update_losses(self, losses, learning_rate, logscale=False):
        super().update_losses(losses)
        # Prepare values
        batches_per_epoch = min(len(self.c.train_loader), self.c.max_batches_per_epoch)
        y = np.array([losses])
        if logscale: y = np.log10(y)
        # if not np.all(np.isfinite(y)): return

        # Add new loss values to plotted trajectories
        traces = [dict(x=[(self.epoch-1) * batches_per_epoch], y=[y[0][i]], name=self.loss_labels[i])
                    for i in range(self.n_losses) if np.isfinite(y[0][i]) and y[0][i] <= 100]
        layout = {'title': f'LR = {learning_rate[0]:.2e}', 'width': 800, 'height': 400, 'margin': {'l': 40, 'r': 0, 'b': 20, 't': 30, 'pad': 0},
                  'ytickmin': -10, 'ytickmax': 10, 'ytickstep': 1}
        self.visdom_handle._send({'data': traces, 'layout': layout, 'win': 'loss_trajectories', 'append': True}, endpoint='update')

    def update_plots(self, latent_sample, x_test):
        # Plot sample from marginal latent distribution
        self.visdom_handle.scatter(X=latent_sample[:,:2], win=self.latent_plot, opts=dict(
                markersize = 3,
                xtickmin = -3, xtickmax = 3, xtickstep = 1,
                ytickmin = -3, ytickmax = 3, ytickstep = 1,
                margin = {'l': 0, 'r': 0, 'b': 0, 't': 0, 'pad': 0}
            ))

        # Update application-specific plot
        if callable(getattr(self.c.data_model, "update_plot", None)):
            self.c.data_model.update_plot(x_test, self.c.vis_y_target)
            self.visdom_handle.matplot(plt.gcf(), win=self.application_plot)

    def update_progress(self, current_batch, current_epoch):
        bgcolor = '#CCCCCC'
        fgcolor = '#20C000'
        batch_progress = current_batch/self.c.max_batches_per_epoch
        epoch_progress = (current_epoch + batch_progress)/self.c.n_epochs
        self.visdom_handle._send({'data': [{
                                               'values': [epoch_progress, 1 - epoch_progress],
                                               'type': 'pie', 'hole': 0.7,
                                               'textinfo': 'none', 'hoverinfo': 'percent',
                                               'marker': {'colors': [fgcolor, bgcolor],
                                                          'line': {'color': bgcolor, 'width': 0}},
                                               'sort': False, 'direction': 'clockwise'
                                           },{
                                               'values': [batch_progress, 1 - batch_progress],
                                               'type': 'pie', 'hole': 0.7,
                                               'domain': {'x': [0.16, 0.84], 'y': [0.16, 0.84]},
                                               'textinfo': 'none', 'hoverinfo': 'percent',
                                               'marker': {'colors': [fgcolor, bgcolor],
                                                          'line': {'color': bgcolor, 'width': 0}},
                                               'sort': False, 'direction': 'clockwise'
                                           }],
                                           'layout': {
                                               'title': f'Epoch {current_epoch:03}/{self.c.n_epochs:03}',
                                               'margin': {'l': 0, 'r': 0, 'b': 20, 't': 30, 'pad': 0},
                                               'annotations': [{'showarrow': False, 'x': 0.5, 'y': 0.5,
                                                                'text': f'Batch<br>{current_batch}',
                                                                'font': {'size': 24}}],
                                               'showlegend': False,
                                           },
                                           'win': 'epochs'})

    def close(self):
        self.visdom_handle.close()


def restart(c, loss_labels):
    global visualizer
    if c.interactive_visualization and visdom.Visdom().check_connection():
        visualizer = LiveVisualizer(c, loss_labels)
    else:
        visualizer = Visualizer(c, loss_labels)
