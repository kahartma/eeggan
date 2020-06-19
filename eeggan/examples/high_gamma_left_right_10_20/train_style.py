#  Author: Kay Hartmann <kg.hartma@gmail.com>

import os

import numpy as np
import torch
from braindecode.torch_ext.modules import IntermediateOutputWrapper
from ignite.engine import Events
from ignite.metrics import MetricUsage
from matplotlib import pyplot
from torch import Tensor, optim
from torch.utils.data import DataLoader

from eeggan.cuda.cuda import to_cuda, init_cuda
from eeggan.data.data import Data
from eeggan.data.high_gamma import load_deeps4
from eeggan.data.high_gamma.dataset import HighGammaDataset
from eeggan.data.preprocess.resample import downsample
from eeggan.examples.high_gamma_left_right_10_20.baseline_style import create_progressive_generator_blocks, \
    create_progressive_discriminator_blocks
from eeggan.examples.high_gamma_left_right_10_20.make_data import DATASET_PATH, DEEP4_PATH
from eeggan.pytorch.utils.weights import weight_filler
from eeggan.training.handlers.metrics import WassersteinMetric, InceptionMetric, FrechetMetric, LossMetric
from eeggan.training.handlers.plots import SpectralPlot
from eeggan.training.progressive.discriminator import ProgressiveDiscriminator
from eeggan.training.progressive.generator import ProgressiveGenerator
from eeggan.training.progressive.handler import ProgressionHandler
from eeggan.training.trainer.gan_softplus import GanSoftplusTrainer

SUBJ_IND = 1
RESULT_PATH = "/home/khartmann/projects/eeggandata/results/%d/style" % SUBJ_IND
PLOT_PATH = os.path.join(RESULT_PATH, "plots")
os.makedirs(PLOT_PATH, exist_ok=True)

n_chans = 21  # number of channels in data
n_classes = 2  # number of classes in data
orig_fs = 512.  # sampling rate of data
final_fs = orig_fs / 2  # reduced sampling rate of data

n_batch = 128  # batch size
n_stages = 6  # number of progressive stages
n_epochs_per_stage = 2000  # epochs in each progressive stage
n_epochs_metrics = 2000
plot_every_epoch = 50
n_epochs_fade = int(0.1 * n_epochs_per_stage)
use_fade = False

n_latent = 200  # latent vector size
lr_d = 0.005  # discriminator learning rate
r1_gamma = 0.
r2_gamma = 0.
lr_g = 0.001  # generator learning rate
betas = (0., 0.99)  # optimizer betas

if __name__ == '__main__':
    init_cuda()  # activate cuda

    dataset = HighGammaDataset(1, DATASET_PATH)  # load dataset

    train_data = dataset.train_data
    test_data = dataset.test_data
    n_time = train_data.X.shape[2]  # number of samples
    n_time_last_layer = int(np.floor(n_time / 2 ** n_stages))  # number of samples in last discriminator layer

    # create discriminator and generator modules
    discriminator = ProgressiveDiscriminator(n_time, n_chans, n_classes,
                                             create_progressive_discriminator_blocks(n_chans, n_time_last_layer,
                                                                                     n_classes))
    generator = ProgressiveGenerator(n_time, n_chans, n_classes, n_latent,
                                     create_progressive_generator_blocks(n_chans, n_latent, n_time_last_layer,
                                                                         n_classes))

    # initiate weights
    generator.apply(weight_filler)
    discriminator.apply(weight_filler)
    generator, discriminator = to_cuda(generator, discriminator)

    # trainer engine
    trainer = GanSoftplusTrainer(10, discriminator, generator, r1_gamma, r2_gamma)

    # handles potential progression after each epoch
    progression_handler = ProgressionHandler(discriminator, generator, n_stages, use_fade, epochs_fade=n_epochs_fade)
    progression_handler.set_progression(0, 1.)
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), progression_handler.advance_alpha)

    # usage to update every epoch and compute once at end of stage
    usage_metrics = MetricUsage(Events.STARTED, Events.EPOCH_COMPLETED(every=n_epochs_per_stage),
                                Events.EPOCH_COMPLETED(every=n_epochs_metrics))

    for stage in range(n_stages):
        # optimizer
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=lr_d, betas=betas)
        optim_generator = optim.Adam(generator.parameters(), lr=lr_g, betas=betas)
        trainer.set_optimizers(optim_discriminator, optim_generator)

        # modules to save
        to_save = {'discriminator': discriminator, 'generator': generator,
                   'optim_discriminator': optim_discriminator, 'optim_generator': optim_generator}

        # load trained deep4s for stage
        deep4s = load_deeps4(SUBJ_IND, stage, DEEP4_PATH)
        select_modules = ['conv_4', 'softmax']
        deep4s = [to_cuda(IntermediateOutputWrapper(select_modules, deep4)) for deep4 in deep4s]

        # scale data for current stagee
        sample_factor = 2 ** (n_stages - stage - 1)
        X_block = downsample(train_data.X, factor=sample_factor, axis=2)

        # initiate spectral plotter
        spectral_plot = SpectralPlot(pyplot.figure(), PLOT_PATH, "spectral_stage_%d_" % stage, X_block.shape[2],
                                     orig_fs / sample_factor)
        event_name = Events.EPOCH_COMPLETED(every=plot_every_epoch)
        spectral_handler = trainer.add_event_handler(event_name, spectral_plot)

        # initiate metrics
        metric_wasserstein = WassersteinMetric(100, np.prod(X_block.shape[1:]).item())
        metric_inception = InceptionMetric(deep4s, sample_factor)
        metric_frechet = FrechetMetric(deep4s, sample_factor)
        metric_loss = LossMetric()
        metrics = [metric_wasserstein, metric_inception, metric_frechet, metric_loss]
        metric_names = ["wasserstein", "inception", "frechet", "loss"]
        trainer.attach_metrics(metrics, metric_names, usage_metrics)

        # wrap into cuda loader
        train_data_tensor = Data[Tensor](*to_cuda(Tensor(X_block), Tensor(train_data.y), Tensor(train_data.y_onehot)))
        train_loader = DataLoader(train_data_tensor, batch_size=n_batch, shuffle=True)

        # train stage
        state = trainer.run(train_loader, (stage + 1) * n_epochs_per_stage)
        trainer.remove_event_handler(spectral_plot, event_name)  # spectral_handler.remove() does not work :(

        # save stuff
        torch.save(to_save, os.path.join(RESULT_PATH, 'modules_stage_%d.pt' % stage))
        torch.save(dict([(name, to_save[name].state_dict()) for name in to_save.keys()]),
                   os.path.join(RESULT_PATH, 'states_stage_%d.pt' % stage))
        torch.save(trainer.state.metrics, os.path.join(RESULT_PATH, 'metrics_stage_%d.pt' % stage))

        # advance stage of not last
        trainer.detach_metrics(metrics, usage_metrics)
        if stage != n_stages - 1:
            progression_handler.advance_stage()
