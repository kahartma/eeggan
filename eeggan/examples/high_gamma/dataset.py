#  Author: Kay Hartmann <kg.hartma@gmail.com>
from typing import List, Tuple

import numpy as np

from eeggan.data.dataset import Data, Dataset


class HighGammaDataset(Dataset[np.ndarray]):
    def __init__(self, train_data: Data[np.ndarray], test_data: Data[np.ndarray], n_time: int, channels: List[str],
                 classes: List[str], fs: float, segment_ival_ms: Tuple[int, int]):
        super().__init__(train_data, test_data)

        self.n_time = n_time
        self.channels = channels
        self.classes = classes
        self.fs = fs
        self.segment_ival_ms = segment_ival_ms
