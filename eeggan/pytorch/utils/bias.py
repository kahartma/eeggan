#  Author: Kay Hartmann <kg.hartma@gmail.com>

def fill_bias_zero(b):
    if b is not None:
        b.data.fill_(0.0)
