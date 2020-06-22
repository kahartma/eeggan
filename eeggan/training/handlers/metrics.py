#  Author: Kay Hartmann <kg.hartma@gmail.com>

from abc import ABCMeta
from typing import List, Tuple, TypeVar, Generic, Dict

import numpy as np
import torch
from ignite.metrics import Metric
from torch import Tensor
from torch.nn.modules.module import Module

from eeggan.cuda import to_device
from eeggan.data.preprocess.resample import upsample
from eeggan.training.trainer.trainer import BatchOutput
from eeggan.validation.metrics.frechet import calculate_activation_statistics, calculate_frechet_distances
from eeggan.validation.metrics.inception import calculate_inception_score
from eeggan.validation.metrics.wasserstein import create_wasserstein_transform_matrix, \
    calculate_sliced_wasserstein_distance
from eeggan.validation.validation_helper import logsoftmax_act_to_softmax

T = TypeVar('T')


class ListMetric(Metric, Generic[T], metaclass=ABCMeta):
    def __init__(self):
        self.values: List[Tuple[int, T]]
        Metric.__init__(self)

    def reset(self) -> None:
        self.values = []

    def append(self, value: Tuple[int, T]):
        self.values.append(value)

    def compute(self) -> List[Tuple[int, T]]:
        return self.values


class WassersteinMetric(ListMetric[float]):

    def __init__(self, n_projections: int, n_features: int):
        self.n_projections = n_projections
        self.n_features = n_features
        self.w_transform: np.ndarray
        super().__init__()

    def reset(self) -> None:
        super().reset()
        self.w_transform = create_wasserstein_transform_matrix(self.n_projections, self.n_features)

    def update(self, batch_output: BatchOutput) -> None:
        epoch = batch_output.i_epoch
        X_real = batch_output.batch_real.X.data.cpu().numpy()
        X_fake = batch_output.batch_fake.X.data.cpu().numpy()
        distance = calculate_sliced_wasserstein_distance(X_real, X_fake, self.w_transform)
        self.append((epoch, distance))


class InceptionMetric(ListMetric[Tuple[float, float]]):

    def __init__(self, deep4s: List[Module], upsample_factor: float, splits: int = 1, repetitions: int = 100):
        self.deep4s = deep4s
        self.upsample_factor = upsample_factor
        self.splits = splits
        self.repetitions = repetitions
        super().__init__()

    def reset(self) -> None:
        super().reset()

    def update(self, batch_output: BatchOutput) -> None:
        X_fake, = to_device(batch_output.batch_fake.X.device,
                            Tensor(
                                upsample(batch_output.batch_fake.X.data.cpu().numpy(), self.upsample_factor, axis=2)))
        X_fake = X_fake[:, :, :, None]
        epoch = batch_output.i_epoch
        score_means = []
        score_stds = []
        for deep4 in self.deep4s:
            with torch.no_grad():
                preds = deep4(X_fake)[1]
                preds = logsoftmax_act_to_softmax(preds)
                score_mean, score_std = calculate_inception_score(preds, self.splits, self.repetitions)
            score_means.append(score_mean)
            score_stds.append(score_std)
        self.append((epoch, (np.mean(score_means).item(), np.mean(score_stds).item())))


class FrechetMetric(ListMetric[Tuple[float, float]]):

    def __init__(self, deep4s: List[Module], upsample_factor: float):
        self.deep4s = deep4s
        self.upsample_factor = upsample_factor
        super().__init__()

    def reset(self) -> None:
        super().reset()

    def update(self, batch_output: BatchOutput) -> None:
        with torch.no_grad():
            X_real, = to_device(batch_output.batch_real.X.device,
                                Tensor(
                                    upsample(batch_output.batch_real.X.data.cpu().numpy(), self.upsample_factor,
                                             axis=2)))
            X_real = X_real[:, :, :, None]
            X_fake, = to_device(batch_output.batch_fake.X.device,
                                Tensor(
                                    upsample(batch_output.batch_fake.X.data.cpu().numpy(), self.upsample_factor,
                                             axis=2)))
            X_fake = X_fake[:, :, :, None]
            epoch = batch_output.i_epoch
            dists = []
            for deep4 in self.deep4s:
                mu_real, sig_real = calculate_activation_statistics(deep4(X_real)[0])
                mu_fake, sig_fake = calculate_activation_statistics(deep4(X_fake)[0])
                dist = calculate_frechet_distances(mu_real[None, :, :], sig_real[None, :, :], mu_fake[None, :, :],
                                                   sig_fake[None, :, :]).item()
                dists.append(dist)
            self.append((epoch, (np.mean(dists).item(), np.std(dists).item())))


class ClassificationMetric(ListMetric[Tuple[float, float]]):

    def __init__(self, deep4s: List[Module], upsample_factor: float):
        self.deep4s = deep4s
        self.upsample_factor = upsample_factor
        super().__init__()

    def reset(self) -> None:
        super().reset()

    def update(self, batch_output: BatchOutput) -> None:
        X_fake, = to_device(batch_output.batch_fake.X.device,
                            Tensor(
                                upsample(batch_output.batch_fake.X.data.cpu().numpy(), self.upsample_factor, axis=2)))
        X_fake = X_fake[:, :, :, None]
        epoch = batch_output.i_epoch
        accuracies = []
        for deep4 in self.deep4s:
            with torch.no_grad():
                preds: Tensor = deep4(X_fake)[1].squeeze()
                class_pred = preds.argmax(dim=1)
                accuracy = (class_pred == batch_output.batch_fake.y).type(torch.float).mean()
                accuracies.append(accuracy.item())
        self.append((epoch, (np.mean(accuracies).item(), np.std(accuracies).item())))


class LossMetric(ListMetric[Dict]):

    def __init__(self):
        super().__init__()

    def reset(self) -> None:
        super().reset()

    def update(self, batch_output: BatchOutput) -> None:
        self.append((batch_output.i_epoch, {"loss_d": batch_output.loss_d, "loss_g": batch_output.loss_g}))
