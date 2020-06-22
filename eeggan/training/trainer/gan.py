#  Author: Kay Hartmann <kg.hartma@gmail.com>

import torch
from torch import sigmoid
from torch.nn import BCELoss

from eeggan.cuda import to_device
from eeggan.data.data import Data
from eeggan.training.discriminator import Discriminator
from eeggan.training.generator import Generator
from eeggan.training.trainer.trainer import Trainer
from eeggan.training.trainer.utils import detach_all


class GanTrainer(Trainer):

    def __init__(self, i_logging, discriminator: Discriminator, generator: Generator):
        self.loss = BCELoss()
        super().__init__(i_logging, discriminator, generator)

    def train_discriminator(self, batch_real: Data[torch.Tensor], batch_fake: Data[torch.Tensor], latent: torch.Tensor):
        self.discriminator.zero_grad()
        self.optim_discriminator.zero_grad()
        self.discriminator.train(True)

        fx_real = sigmoid(self.discriminator(batch_real.X.requires_grad_(False), y=batch_real.y.requires_grad_(False),
                                             y_onehot=batch_real.y_onehot.requires_grad_(False)))
        loss_real = self.loss(fx_real, torch.ones_like(fx_real))
        loss_real.backward()

        fx_fake = sigmoid(self.discriminator(batch_fake.X.requires_grad_(False), y=batch_fake.y.requires_grad_(False),
                                             y_onehot=batch_fake.y_onehot.requires_grad_(False)))
        loss_fake = self.loss(fx_fake, torch.zeros_like(fx_real))
        loss_fake.backward()

        self.optim_discriminator.step()

        return {"loss_real": loss_real.item(), "loss_fake": loss_fake.item()}

    def train_generator(self, batch_real: Data[torch.Tensor]):
        self.generator.zero_grad()
        self.optim_generator.zero_grad()
        self.generator.train(True)
        self.discriminator.train(False)

        with torch.no_grad():
            latent, y_fake, y_onehot_fake = to_device(batch_real.X.device,
                                                      *self.generator.create_latent_input(self.rng, len(batch_real.X)))
            latent, y_fake, y_onehot_fake = detach_all(latent, y_fake, y_onehot_fake)

        X_fake = self.generator(latent.requires_grad_(False), y=y_fake.requires_grad_(False),
                                y_onehot=y_onehot_fake.requires_grad_(False))

        batch_fake = Data[torch.Tensor](X_fake, y_fake, y_onehot_fake)
        fx_fake = sigmoid(self.discriminator(batch_fake.X.requires_grad_(True), y=batch_fake.y.requires_grad_(True),
                                             y_onehot=batch_fake.y_onehot.requires_grad_(True)))
        loss = self.loss(fx_fake, torch.ones_like(fx_fake))
        loss.backward()

        self.optim_generator.step()

        return loss.item()
