#  Author: Kay Hartmann <kg.hartma@gmail.com>
import torch
from torch import autograd

from eeggan.cuda.cuda import to_device
from eeggan.data.data import Data
from eeggan.training.discriminator import Discriminator
from eeggan.training.trainer.trainer import Trainer


class WganGpTrainer(Trainer):

    def __init__(self, i_logging, discriminator: Discriminator, generator: torch.Generator, lambd: float,
                 one_sided_penalty: bool, distance_weighting: bool,
                 eps_drift: float, eps_center: float):
        self.lambd = lambd
        self.one_sided_penalty = one_sided_penalty
        self.distance_weighting = distance_weighting
        self.eps_drift = eps_drift
        self.eps_center = eps_center
        super().__init__(i_logging, discriminator, generator)

    def train_discriminator(self, batch_real: Data[torch.Tensor], batch_fake: Data[torch.Tensor], latent: torch.Tensor):
        self.discriminator.zero_grad()
        self.optim_discriminator.zero_grad()
        self.discriminator.train(True)

        fx_real = self.discriminator(batch_real.X.requires_grad_(False), y=batch_real.y.requires_grad_(False),
                                     y_onehot=batch_real.y_onehot.requires_grad_(False))
        loss_real = -fx_real.mean()
        loss_real.backward(retain_graph=(self.eps_drift > 0 or self.eps_center > 0))

        fx_fake = self.discriminator(batch_fake.X.requires_grad_(False), y=batch_fake.y.requires_grad_(False),
                                     y_onehot=batch_fake.y_onehot.requires_grad_(False))
        loss_fake = fx_fake.mean()
        loss_fake.backward(retain_graph=(self.eps_drift > 0 or self.eps_center > 0))

        loss_drift = None
        loss_center = None
        if self.eps_drift > 0:
            loss_drift = self.eps_drift * loss_real ** 2
            loss_drift.backward(retain_graph=self.eps_center > 0)
            loss_drift = loss_drift.item()
        if self.eps_center > 0:
            loss_center = (loss_real + loss_fake)
            loss_center = self.eps_center * loss_center ** 2
            loss_center.backward()
            loss_center = loss_center.item()

        loss_penalty = None
        if self.lambd > 0.:
            loss_penalty = self.calc_gradient_penalty(batch_real.X.requires_grad_(True),
                                                      batch_fake.X.requires_grad_(True),
                                                      batch_real.y_onehot.requires_grad_(True),
                                                      batch_fake.y_onehot.requires_grad_(True))
            dist = 1
            if self.distance_weighting:
                dist = (loss_real - loss_fake).detach()
                dist = dist.clamp(min=0)

            loss_penalty = self.lambd * dist * loss_penalty
            loss_penalty.backward()
            loss_penalty = loss_penalty.item()

        self.optim_discriminator.step()

        return {"loss_real": loss_real.item(), "loss_fake": loss_fake.item(), "gp": loss_penalty,
                "drift_penalty": loss_drift, "center_penalty": loss_center}

    def train_generator(self, batch_real: Data[torch.Tensor]):
        self.generator.zero_grad()
        self.optim_generator.zero_grad()
        self.generator.train(True)
        self.discriminator.train(False)

        latent, y_fake, y_onehot_fake = to_device(batch_real.X.device,
                                                  *self.generator.create_latent_input(self.rng, len(batch_real.X)))
        X_fake = self.generator(latent.requires_grad_(False), y=y_fake.requires_grad_(False),
                                y_onehot=y_onehot_fake.requires_grad_(False))
        batch_fake = Data[torch.Tensor](X_fake, y_fake, y_onehot_fake)

        fx_fake = self.discriminator(batch_fake.X.requires_grad_(True), y=batch_fake.y.requires_grad_(True),
                                     y_onehot=batch_fake.y_onehot.requires_grad_(True))
        loss = fx_fake.mean()
        loss.backward()
        self.optim_generator.step()

        return loss.item()

    def calc_gradient_penalty(self, X_real: torch.Tensor, X_fake: torch.Tensor, y_real: torch.Tensor,
                              y_fake: torch.Tensor):
        """
        Improved WGAN gradient penalty
        """
        alpha_tmp = torch.rand(X_real.size(0)).to(X_real)
        alpha_X = alpha_tmp[:, None, None].expand_as(X_real)
        interpolates = alpha_X * X_real + ((1 - alpha_X) * X_fake)

        interpolates_y = None
        if y_real is not None and y_fake is not None:
            alpha_y = alpha_tmp[:, None].expand_as(y_real)
            interpolates_y = alpha_y * y_real + ((1 - alpha_y) * y_fake)

        disc_interpolates = self.discriminator(interpolates.requires_grad_(True),
                                               y_onehot=interpolates_y.requires_grad_(True))
        ones = torch.ones_like(disc_interpolates)

        if interpolates_y is not None:
            interpolates = (interpolates, interpolates_y)
            disc_interpolates = (disc_interpolates, disc_interpolates)
            ones = (ones, ones)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=ones,
                                  create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)

        gradients = torch.cat([grad.reshape(grad.size(0), -1) for grad in gradients], dim=1)

        tmp = gradients.norm(2, dim=1) - 1
        if self.one_sided_penalty:
            tmp = tmp.clamp(min=0)
        gradient_penalty = (tmp ** 2).mean()

        return gradient_penalty
