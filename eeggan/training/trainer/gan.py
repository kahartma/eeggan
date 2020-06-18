#  Author: Kay Hartmann <kg.hartma@gmail.com>

from typing import List, Tuple

import torch
from ignite.metrics import Metric, MetricUsage
from torch import sigmoid
from torch.nn import BCELoss
from torch.optim.optimizer import Optimizer

from eeggan.data.data import Data
from eeggan.training.discriminator import Discriminator
from eeggan.training.generator import Generator
from eeggan.training.trainer.trainer import Trainer


class GanTrainer(Trainer):

    def __init__(self, discriminator: Discriminator, generator: Generator, i_logging,
                 metrics: List[Tuple[Metric, MetricUsage]], optimizer_disc: Optimizer, optimizer_gen: Optimizer):
        self.loss = BCELoss()
        self.optimizer_disc = optimizer_disc
        self.optimizer_gen = optimizer_gen
        super().__init__(discriminator, generator, i_logging, metrics)

    def train_discriminator(self, batch_real: Data[torch.Tensor], batch_fake: Data[torch.Tensor], latent: torch.Tensor):
        self.discriminator.zero_grad()
        self.optimizer_disc.zero_grad()
        fx_real = sigmoid(self.discriminator(batch_real.X, y=batch_real.y, y_onehot=batch_real.y_onehot))
        loss_real = self.loss(fx_real, torch.ones_like(fx_real))
        loss_real.backward()
        fx_fake = sigmoid(self.discriminator(batch_fake.X, y=batch_fake.y, y_onehot=batch_fake.y_onehot))
        loss_fake = self.loss(fx_fake, torch.zeros_like(fx_real))
        loss_fake.backward()
        self.optimizer_disc.step()
        return loss_real.data.item(), loss_fake.data.item()

    def train_generator(self, batch_real: Data[torch.Tensor], batch_fake: Data[torch.Tensor], latent: torch.Tensor):
        self.generator.zero_grad()
        self.optimizer_gen.zero_grad()
        train_tmp = self.discriminator.training
        self.discriminator.eval()
        fx_fake = self.discriminator(batch_fake.X, y=batch_fake.y, y_onehot=batch_fake.y_onehot)
        self.discriminator.train(train_tmp)
        loss = self.loss(fx_fake, torch.ones_like(fx_fake))
        loss.backward()
        self.optimizer_gen.step()
        return loss.item()
