#  Author: Kay Hartmann <kg.hartma@gmail.com>

def normalize_data(x):
    x = x - x.mean()
    x = x / x.std()
    return x
