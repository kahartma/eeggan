#  Author: Kay Hartmann <kg.hartma@gmail.com>


from eeggan.cuda import init_cuda

# https://gin.g-node.org/robintibor/high-gamma-dataset/
from eeggan.examples.high_gamma.make_data import make_dataset_for_subj, CHANNELS_10_20, \
    CLASSDICT_REST_RIGHT_HAND, make_deep4_for_subj

FS = 512.
SEGMENT_IVAL = (500, 2250)
INPUT_LENGTH = int(1.75 * FS)
N_PROGRESSIVE_STAGES = 6
N_DEEP4 = 10


def run(subj_ind: int, high_gamma_datapath: str, dataset_path: str, deep4_path: str):
    init_cuda()  # activate cuda

    make_dataset_for_subj(subj_ind, high_gamma_datapath, dataset_path, CHANNELS_10_20, CLASSDICT_REST_RIGHT_HAND,
                          FS, SEGMENT_IVAL, INPUT_LENGTH)
    make_deep4_for_subj(subj_ind, dataset_path, deep4_path, N_PROGRESSIVE_STAGES, N_DEEP4)
