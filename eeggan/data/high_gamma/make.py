#  Author: Kay Hartmann <kg.hartma@gmail.com>

import copy
import os
from collections import OrderedDict

import joblib
import mne
import numpy as np
from braindecode.datautil.signalproc import exponential_running_standardize
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.mne_ext.signalproc import resample_cnt, mne_apply

from eeggan.data.preprocess.resample import upsample, downsample
from eeggan.data.preprocess.util import prepare_data
from eeggan.examples.high_gamma_left_right_10_20.braindecode_hack import BBCIDataset
from eeggan.validation.deep4 import train_completetrials

SUBJ_INDECES = np.arange(1, 15)
CHANNELS_10_20 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4',
                  'P8', 'O1', 'O2', 'M1', 'M2']

CLASSDICT_ALL = OrderedDict([('Right', 1), ('Left', 2), ('Rest', 3), ('Feet', 4)])
CLASSDICT_RIGHT_LEFT_HAND = OrderedDict([('Right', 1), ('Left', 2)])
CLASSDICT_REST_RIGHT_HAND = OrderedDict([('Rest', 3), ('Right', 1)])


def make_dataset_for_subj(subj_ind, highgamma_path, dataset_path, channels, classdict, fs, segment_ival_ms,
                          input_length):
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    n_classes = len(classdict)
    train_filename = os.path.join(highgamma_path, 'train', '%s.mat' % subj_ind)
    train_set = extract_dataset(train_filename, channels, classdict, fs, segment_ival_ms)
    train_set = prepare_data(train_set.X, train_set.y, n_classes, input_length, normalize=True)

    test_filename = os.path.join(highgamma_path, 'test', '%s.mat' % subj_ind)
    test_set = extract_dataset(test_filename, channels, classdict, fs, segment_ival_ms)
    test_set = prepare_data(test_set.X, test_set.y, n_classes, input_length, normalize=True)

    dump = dict(train_set=train_set, test_set=test_set, channels=channels, classes=classdict, fs=fs,
                segment_ival_ms=segment_ival_ms)
    joblib.dump(dump, os.path.join(dataset_path, '%s.dataset' % subj_ind), compress=True)


def make_deep4_for_subj(subj_ind, dataset_path, deep4_path, n_progressive, n_deep4, fs):
    if not os.path.exists(deep4_path):
        os.makedirs(deep4_path)

    dataset = load_dataset(subj_ind, dataset_path)
    for i_stage in np.arange(n_progressive):
        models = []
        train_set_stage = copy.copy(dataset['train_set'])
        test_set_stage = copy.copy(dataset['test_set'])
        train_set_stage.X = make_data_for_stage(train_set_stage.X, i_stage, n_progressive - 1)
        test_set_stage.X = make_data_for_stage(test_set_stage.X, i_stage, n_progressive - 1)
        for i in range(n_deep4):
            exp = make_deep4(train_set_stage, test_set_stage, len(dataset['classes']))
            models.append(exp.model)

        joblib.dump(models, os.path.join(deep4_path, '%s_stage%s.deep4' % (subj_ind, i_stage)), compress=True)


def make_data_for_stage(X, i_stage, max_stage):
    down = downsample(X, 2 ** (max_stage - i_stage), axis=2)
    up = upsample(down, 2 ** (max_stage - i_stage), axis=2)
    return downsample(up, 2)


def make_deep4(train_set, test_set, n_classes):
    exp = train_completetrials(train_set, test_set, n_classes, max_epochs=25, batch_size=60, cuda=True)
    exp.model = exp.model.cpu().eval()
    return exp


def load_dataset(index, path):
    return joblib.load(os.path.join(path, '%s.dataset' % index))


def load_deeps4(index, stage, path):
    return joblib.load(os.path.join(path, '%s_stage%s.deep4' % (index, stage)))


def extract_dataset(filename, channels, classdict, fs, segment_ival_ms):
    cnt = load_mne_dataset(filename)
    cnt = pick_channels(cnt, channels)
    cnt = car_signal(cnt)
    cnt = resample(cnt, fs)
    cnt = standardize(cnt)

    dataset = get_signal_from_cnt(cnt, classdict, segment_ival_ms)
    return dataset


def load_mne_dataset(filename):
    return BBCIDataset(filename).load()


def car_signal(cnt):
    tmp_info = mne.create_info(ch_names=cnt.info['ch_names'], sfreq=cnt.info['sfreq'], ch_types='eeg')
    cnt.info['chs'] = tmp_info['chs']
    cnt.set_eeg_reference('average', projection=False)
    return cnt


def pick_channels(cnt, channels):
    return cnt.pick_channels(channels)


def resample(cnt, fs):
    return resample_cnt(cnt, fs)


def standardize(cnt):
    return mne_apply(lambda a: exponential_running_standardize(
        a.T, init_block_size=1000, factor_new=0.001, eps=1e-4).T,
                     cnt)


def get_signal_from_cnt(cnt, markers, interval):
    return create_signal_target_from_raw_mne(cnt, markers, interval)
