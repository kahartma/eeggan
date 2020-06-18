#  Author: Kay Hartmann <kg.hartma@gmail.com>

import numpy as np
import scipy.signal as sig


def resample(x, ofs, nfs, axis=0):
    # beta value taken from https://github.com/bmcfee/resampy/blob/master/resampy/filters.py resampy seems bugged
    up = 1
    down = 1
    if ofs < nfs:
        up = float(nfs) / ofs
    elif ofs > nfs:
        down = float(ofs) / nfs
    return sig.resample_poly(x, up, down, axis=axis, window=('kaiser', 14.769656459379492)).astype(
        np.float32)


def upsample(x, factor, axis=0):
    return resample(x, 1, factor, axis=axis)


def downsample(x, factor, axis=0):
    return resample(x, factor, 1, axis=axis)
