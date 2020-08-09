#  Author: Kay Hartmann <kg.hartma@gmail.com>
from ignite.engine import Events

from eeggan.examples.high_gamma.high_gamma_rest_right_10_20.make_data import DATASET_PATH, DEEP4_PATH, INPUT_LENGTH, FS, \
    N_PROGRESSIVE_STAGES
from eeggan.examples.high_gamma.models.baseline import Baseline
from eeggan.examples.high_gamma.train import train
from eeggan.pytorch.utils.weights import weight_filler
from eeggan.training.progressive.handler import ProgressionHandler
from eeggan.training.trainer.gan_softplus import GanSoftplusTrainer

SUBJ_IND = 1
RESULT_PATH = "/home/khartmann/projects/eeggandata/results/%d/train" % SUBJ_IND

n_chans = 21  # number of channels in data
n_classes = 2  # number of classes in data
orig_fs = FS  # sampling rate of data

n_batch = 128  # batch size
n_stages = N_PROGRESSIVE_STAGES  # number of progressive stages
n_epochs_per_stage = 2000  # epochs in each progressive stage
n_epochs_metrics = 100
plot_every_epoch = 100
n_epochs_fade = int(0.1 * n_epochs_per_stage)
use_fade = False

n_latent = 200  # latent vector size
lr_d = 0.005  # discriminator learning rate
r1_gamma = 0.
r2_gamma = 0.
lr_g = 0.001  # generator learning rate
betas = (0., 0.99)  # optimizer betas

n_filters = 120
n_time = INPUT_LENGTH

if __name__ == '__main__':
    # create discriminator and generator modules
    model_builder = Baseline(n_stages, n_latent, n_time, n_chans, n_classes, n_filters)
    discriminator = model_builder.build_discriminator()
    generator = model_builder.build_generator()

    # initiate weights
    generator.apply(weight_filler)
    discriminator.apply(weight_filler)

    # trainer engine
    trainer = GanSoftplusTrainer(10, discriminator, generator, r1_gamma, r2_gamma)

    # handles potential progression after each epoch
    progression_handler = ProgressionHandler(discriminator, generator, n_stages, use_fade, n_epochs_fade)
    progression_handler.set_progression(0, 1.)
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), progression_handler.advance_alpha)

    train(SUBJ_IND, DATASET_PATH, DEEP4_PATH, RESULT_PATH, progression_handler, trainer, n_batch, lr_d, lr_g, betas,
          n_epochs_per_stage, n_epochs_metrics, plot_every_epoch, orig_fs)
