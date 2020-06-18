#  Author: Kay Hartmann <kg.hartma@gmail.com>

import torch
from torch import autograd
from torch.nn.functional import softplus
from torch.optim.optimizer import Optimizer

from eeggan.data.data import Data
from eeggan.training.discriminator import Discriminator
from eeggan.training.generator import Generator
from eeggan.training.trainer.trainer import Trainer


class GanSoftplusTrainer(Trainer):

    def __init__(self, discriminator: Discriminator, generator: Generator, r1_gamma: float, r2_gamma: float,
                 i_logging: int, optimizer_disc: Optimizer, optimizer_gen: Optimizer):
        self.r1_gamma = r1_gamma
        self.r2_gamma = r2_gamma
        self.optimizer_disc = optimizer_disc
        self.optimizer_gen = optimizer_gen
        super().__init__(discriminator, generator, i_logging)

    def train_discriminator(self, batch_real: Data[torch.Tensor], batch_fake: Data[torch.Tensor], latent: torch.Tensor):
        self.discriminator.zero_grad()
        self.optimizer_disc.zero_grad()

        # Compute output and loss
        X_real = batch_real.X.requires_grad_()
        y_real = batch_real.y.requires_grad_()
        y_onehot_real = batch_real.y_onehot.requires_grad_()
        fx_real = self.discriminator(X_real, y=y_real, y_onehot=y_onehot_real)
        loss_real = softplus(-fx_real).mean()
        loss_real.backward(retain_graph=self.r1_gamma > 0)
        loss_r1 = None
        if self.r1_gamma > 0:
            r1_penalty = self.r1_gamma * calc_gradient_penalty(X_real, y_onehot_real, fx_real)
            r1_penalty.backward()
            loss_r1 = r1_penalty.item()

        X_fake = batch_real.X.requires_grad_()
        y_fake = batch_real.y.requires_grad_()
        y_onehot_fake = batch_real.y_onehot.requires_grad_()
        fx_fake = self.discriminator(X_fake, y=y_fake, y_onehot=y_onehot_fake)
        loss_fake = softplus(fx_fake).mean()
        loss_fake.backward(retain_graph=self.r2_gamma > 0)
        loss_r2 = None
        if self.r2_gamma > 0:
            r2_penalty = self.r2_gamma * calc_gradient_penalty(X_fake, y_onehot_fake, fx_fake)
            r2_penalty.backward()
            loss_r2 = r2_penalty.item()

        self.optimizer_disc.step()
        return loss_real.data.item(), loss_fake.data.item(), loss_r1, loss_r2

    def train_generator(self, batch_real: Data[torch.Tensor], batch_fake: Data[torch.Tensor], latent: torch.Tensor):
        for p in self.discriminator.parameters():
            p.requires_grad = False
        self.generator.zero_grad()
        self.optimizer_gen.zero_grad()
        fx_fake = self.discriminator(batch_fake.X, y=batch_fake.y, y_onehot=batch_fake.y_onehot)
        loss = softplus(-fx_fake).mean()
        loss.backward()
        self.optimizer_gen.step()
        for p in self.discriminator.parameters():
            p.requires_grad = True
        return loss.item()


def calc_gradient_penalty(X: torch.Tensor, y: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    inputs = X
    ones = torch.ones_like(outputs)
    if y is not None:
        inputs = (inputs, y)
        outputs = (outputs, outputs)
        ones = (ones, ones)
    gradients = autograd.grad(outputs=outputs, inputs=inputs,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)
    gradients = torch.cat([tmp.reshape(tmp.size(0), -1) for tmp in gradients], 1)
    gradient_penalty = 0.5 * gradients.norm(2, dim=1).pow(2).mean()
    return gradient_penalty
