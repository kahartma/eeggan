#  Author: Kay Hartmann <kg.hartma@gmail.com>

import torch
from torch import autograd
from torch.nn.functional import softplus

from eeggan.cuda import to_device
from eeggan.data.data import Data
from eeggan.training.discriminator import Discriminator
from eeggan.training.generator import Generator
from eeggan.training.trainer.trainer import Trainer
from eeggan.training.trainer.utils import detach_all


class GanSoftplusTrainer(Trainer):
    """
    Improved GAN

    References
    ----------
    Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., &
    Chen, X. (2016). Improved Techniques for Training GANs. Learning;
    Computer Vision and Pattern Recognition; Neural and Evolutionary Computing.
    Retrieved from http://arxiv.org/abs/1606.03498
    """

    def __init__(self, i_logging: int, discriminator: Discriminator, generator: Generator, r1_gamma: float,
                 r2_gamma: float):
        self.r1_gamma = r1_gamma
        self.r2_gamma = r2_gamma
        super().__init__(i_logging, discriminator, generator)

    def train_discriminator(self, batch_real: Data[torch.Tensor], batch_fake: Data[torch.Tensor], latent: torch.Tensor):
        self.discriminator.zero_grad()
        self.optim_discriminator.zero_grad()
        self.discriminator.train(True)

        has_r1 = self.r1_gamma > 0.
        fx_real = self.discriminator(batch_real.X.requires_grad_(has_r1), y=batch_real.y.requires_grad_(has_r1),
                                     y_onehot=batch_real.y_onehot.requires_grad_(has_r1))
        loss_real = softplus(-fx_real).mean()
        loss_real.backward(retain_graph=has_r1)
        loss_r1 = None
        if has_r1:
            r1_penalty = self.r1_gamma * calc_gradient_penalty(batch_real.X.requires_grad_(True),
                                                               batch_real.y_onehot.requires_grad_(True), fx_real)
            r1_penalty.backward()
            loss_r1 = r1_penalty.item()

        has_r2 = self.r2_gamma > 0.
        fx_fake = self.discriminator(batch_fake.X.requires_grad_(has_r2), y=batch_fake.y.requires_grad_(has_r2),
                                     y_onehot=batch_fake.y_onehot.requires_grad_(has_r2))
        loss_fake = softplus(fx_fake).mean()
        loss_fake.backward(retain_graph=has_r2)
        loss_r2 = None
        if has_r2:
            r2_penalty = self.r1_gamma * calc_gradient_penalty(batch_fake.X.requires_grad_(True),
                                                               batch_fake.y_onehot.requires_grad_(True), fx_real)
            r2_penalty.backward()
            loss_r2 = r2_penalty.item()

        self.optim_discriminator.step()

        return {"loss_real": loss_real.item(), "loss_fake": loss_fake.item(), "r1_penalty": loss_r1,
                "r2_penalty": loss_r2}

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

        fx_fake = self.discriminator(batch_fake.X.requires_grad_(True), y=batch_fake.y.requires_grad_(True),
                                     y_onehot=batch_fake.y_onehot.requires_grad_(True))
        loss = softplus(-fx_fake).mean()
        loss.backward()

        self.optim_generator.step()

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
