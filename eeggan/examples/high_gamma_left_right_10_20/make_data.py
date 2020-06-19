#  Author: Kay Hartmann <kg.hartma@gmail.com>

from eeggan.cuda.cuda import init_cuda
from eeggan.data.high_gamma import make_deep4_for_subj, make_dataset_for_subj, CHANNELS_10_20, \
    CLASSDICT_REST_RIGHT_HAND, SUBJ_INDECES

# https://gin.g-node.org/robintibor/high-gamma-dataset/
HIGH_GAMMA_DATAPATH = "/home/khartmann/projects/high-gamma-dataset/data"

# where preprocessed datasets should be saved
DATASET_PATH = "/home/khartmann/projects/eeggandata/datasets"
# where trained deep4s should be saved
DEEP4_PATH = "/home/khartmann/projects/eeggandata/deep4s"

FS = 512.
SEGMENT_IVAL = [500, 2250]
INPUT_LENGTH = int(1.75 * FS)
N_PROGRESSIVE_STAGES = 6
N_DEEP4 = 10

if __name__ == '__main__':
    init_cuda()  # activate cuda

    for subj_ind in SUBJ_INDECES:
        make_dataset_for_subj(subj_ind, HIGH_GAMMA_DATAPATH, DATASET_PATH, CHANNELS_10_20, CLASSDICT_REST_RIGHT_HAND,
                              FS, SEGMENT_IVAL, INPUT_LENGTH)
        make_deep4_for_subj(subj_ind, DATASET_PATH, DEEP4_PATH, N_PROGRESSIVE_STAGES, N_DEEP4, FS)
