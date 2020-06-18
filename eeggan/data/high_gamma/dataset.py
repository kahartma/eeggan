#  Author: Kay Hartmann <kg.hartma@gmail.com>

from eeggan.data.high_gamma import load_dataset


class HighGammaDataset:
    def __init__(self, subj_indx, path):
        dataset = load_dataset(subj_indx, path)
        self.train_data = dataset["train_set"]
        self.test_data = dataset["test_set"]
