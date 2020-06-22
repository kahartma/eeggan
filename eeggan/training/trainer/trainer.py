#  Author: Kay Hartmann <kg.hartma@gmail.com>

from abc import ABCMeta
from typing import List

import torch
from ignite.engine import Engine, Events
from ignite.metrics import Metric, MetricUsage
from numpy.random.mtrand import RandomState
from torch.optim.optimizer import Optimizer

from eeggan.cuda import to_device
from eeggan.data.data import Data
from eeggan.training.discriminator import Discriminator
from eeggan.training.generator import Generator
from eeggan.training.trainer.utils import detach_all


class BatchOutput:
    def __init__(self, i_iteration: int, i_epoch: int, batch_real: Data[torch.Tensor], batch_fake: Data[torch.Tensor],
                 latent: torch.Tensor, loss_d,
                 loss_g):
        self.i_iteration = i_iteration
        self.i_epoch = i_epoch
        self.batch_real = batch_real
        self.batch_fake = batch_fake
        self.latent = latent
        self.loss_d = loss_d
        self.loss_g = loss_g


class Trainer(Engine, metaclass=ABCMeta):
    def __init__(
            self,
            i_logging: int,
            discriminator: Discriminator,
            generator: Generator,
            rng: RandomState = RandomState()
    ):
        self.discriminator = discriminator
        self.generator = generator
        self.rng = rng
        self.optim_discriminator: Optimizer = None
        self.optim_generator: Optimizer = None
        Engine.__init__(self, self.train_batch)
        self.add_event_handler(Events.ITERATION_COMPLETED(every=i_logging), self.log_training)

    def set_optimizers(self, optim_discriminator: Optimizer, optim_generator: Optimizer):
        self.optim_discriminator = optim_discriminator
        self.optim_generator = optim_generator

    def attach_metrics(self, metrics: List[Metric], metric_names: List[str], usage: MetricUsage):
        for i, metric in enumerate(metrics):
            metric.attach(self, metric_names[i], usage=usage)

    def detach_metrics(self, metrics: List[Metric], usage: MetricUsage):
        for metric in metrics:
            metric.detach(self, usage)

    def train_batch(self, engine, batch) -> BatchOutput:
        with torch.no_grad():
            batch_real = Data[torch.Tensor](batch[0], batch[1], batch[2])

            latent, y_fake, y_onehot_fake = to_device(batch_real.X.device,
                                                      *self.generator.create_latent_input(self.rng, len(batch_real.X)))
            X_fake = self.generator(latent, y=y_fake, y_onehot=y_onehot_fake)
            batch_fake = Data[torch.Tensor](X_fake, y_fake, y_onehot_fake)

        batch_real, batch_fake, latent = detach_all(batch_real, batch_fake, latent)
        loss_d = self.train_discriminator(batch_real, batch_fake, latent)
        batch_real = detach_all(batch_real)
        loss_g = self.train_generator(batch_real)

        with torch.no_grad():
            latent, y_fake, y_onehot_fake = to_device(batch_real.X.device,
                                                      *self.generator.create_latent_input(self.rng, len(batch_real.X)))
            X_fake = self.generator(latent, y=y_fake, y_onehot=y_onehot_fake)
            batch_fake = Data[torch.Tensor](X_fake, y_fake, y_onehot_fake)

        batch_real, batch_fake, latent = detach_all(batch_real, batch_fake, latent)
        return BatchOutput(engine.state.iteration, engine.state.epoch, batch_real, batch_fake, latent, loss_d, loss_g)

    def log_training(self, engine):
        batch_out: BatchOutput = engine.state.output
        e = engine.state.epoch
        n = engine.state.max_epochs
        i = engine.state.iteration
        print("Epoch {}/{} : {} - loss_d: {} loss_g: {}".format(e, n, i, batch_out.loss_d, batch_out.loss_g))

    def train_discriminator(self, batch_real: Data[torch.Tensor], batch_fake: Data[torch.Tensor], latent: torch.Tensor):
        raise NotImplementedError

    def train_generator(self, batch_real: Data[torch.Tensor]):
        raise NotImplementedError
