#  Author: Kay Hartmann <kg.hartma@gmail.com>

import numpy as np
import torch.nn.functional as F
from braindecode.datautil.iterators import BalancedBatchSizeIterator, get_balanced_batches
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor
from braindecode.experiments.stopcriteria import MaxEpochs
from braindecode.models.deep4 import Deep4Net
from braindecode.models.util import to_dense_prediction_model
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.optimizers import AdamW
from braindecode.torch_ext.schedulers import CosineAnnealing, ScheduledOptimizer
from numpy.random.mtrand import RandomState


def build_model(input_time_length, n_chans, n_classes, cropped=False):
    model = Deep4Net(n_chans, n_classes, input_time_length=input_time_length,
                     final_conv_length='auto')

    if cropped:
        final_conv_length = model.final_conv_length
        model = Deep4Net(n_chans, n_classes, input_time_length=input_time_length,
                         final_conv_length=final_conv_length)

    model = model.create_network()
    if cropped:
        to_dense_prediction_model(model)

    return model


def train_completetrials(train_set, test_set, n_classes, max_epochs=100, batch_size=60, iterator=None, cuda=True):
    model = build_model(train_set.X.shape[2], int(train_set.X.shape[1]), n_classes, cropped=False)
    if iterator is None:
        iterator = BalancedBatchSizeIterator(batch_size=batch_size, seed=np.random.randint(9999999))
    monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]
    loss_function = F.nll_loss

    return train(train_set, test_set, model, iterator, monitors, loss_function, max_epochs, cuda)


def train(train_set, test_set, model, iterator, monitors, loss_function, max_epochs, cuda):
    if cuda:
        model.cuda()

    optimizer = AdamW(model.parameters(), lr=1 * 0.01,
                      weight_decay=0.5 * 0.001)  # these are good values for the deep model

    stop_criterion = MaxEpochs(max_epochs)
    model_constraint = MaxNormDefaultConstraint()

    n_updates_per_epoch = sum(
        [1 for _ in iterator.get_batches(train_set, shuffle=True)])
    n_updates_per_period = n_updates_per_epoch * max_epochs
    scheduler = CosineAnnealing(n_updates_per_period)
    optimizer = ScheduledOptimizer(scheduler, optimizer, schedule_weight_decay=True)

    exp = Experiment(model, train_set, None, test_set, iterator=iterator,
                     loss_function=loss_function, optimizer=optimizer,
                     model_constraint=model_constraint,
                     monitors=monitors,
                     remember_best_column=None,
                     stop_criterion=stop_criterion,
                     cuda=cuda, run_after_early_stop=False, do_early_stop=False)
    exp.run()
    return exp


class BalancedBatchSizeWithGeneratorIterator(object):

    def __init__(self, batch_size, trial_generator, ratio=0.05, seed=328774):
        self.batch_size = batch_size
        self.trial_generator = trial_generator
        self.ratio = ratio
        self.seed = seed
        self.rng = RandomState(self.seed)

    def get_batches(self, dataset, shuffle):
        n_trials = dataset.X.shape[0]
        batch_size_fake = int(np.ceil(self.batch_size * self.ratio))
        batch_size_real = self.batch_size - batch_size_fake
        batches = get_balanced_batches(n_trials,
                                       batch_size=batch_size_real,
                                       rng=self.rng,
                                       shuffle=shuffle)
        for batch_inds in batches:
            batch_X = dataset.X[batch_inds]
            batch_y = dataset.y[batch_inds]

            fake_X, fake_y = self.trial_generator(batch_size_fake)
            batch_X = np.concatenate((batch_X, fake_X))
            batch_y = np.concatenate((batch_y, fake_y))

            # add empty fourth dimension if necessary
            if batch_X.ndim == 3:
                batch_X = batch_X[:, :, :, None]
            yield (batch_X, batch_y)

    def reset_rng(self):
        self.rng = RandomState(self.seed)
