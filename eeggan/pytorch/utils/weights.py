#  Author: Kay Hartmann <kg.hartma@gmail.com>

def weight_filler(m):
    classname = m.__class__.__name__
    if classname.find('MultiConv') != -1:
        for conv in m.convs:
            conv.weight.data.normal_(0.0, 1.)
            if conv.bias is not None:
                conv.bias.data.fill_(0.)
    elif classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 1.)  # From progressive GAN paper
        if m.bias is not None:
            m.bias.data.fill_(0.)
    elif classname.find('BatchNorm') != -1 or classname.find('LayerNorm') != -1:
        if m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.)


def fill_weights_normal(w):
    if w is not None:
        w.data.normal_(0.0, 1.)
