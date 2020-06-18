#  Author: Kay Hartmann <kg.hartma@gmail.com>

from eeggan.pytorch.modules.module import Module


class PixelShuffle1d(Module):
    """
    1d pixel shuffling
    Shuffles filter dimension into trailing dimension

    Parameters
    ----------
    scale_kernel : int
        Factor of how many filters are shuffled

    References
    ----------
    Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A. P., Bishop, R.,
    … Wang, Z. (2016).
    Real-Time Single Image and Video Super-Resolution Using an Efficient
    Sub-Pixel Convolutional Neural Network.
    Retrieved from http://arxiv.org/abs/1609.05158
    """

    def __init__(self, scale_kernel):
        super(PixelShuffle1d, self).__init__()
        self.scale_kernel = scale_kernel

    def forward(self, x, **kwargs):
        batch_size, channels, in_height = x.size()
        channels /= self.scale_kernel[0]

        out_height = in_height * self.scale_kernel[0]

        input_view = x.contiguous().view(
            batch_size, channels, self.scale_kernel[0], in_height)

        shuffle_out = input_view.permute(0, 1, 3, 2).contiguous()
        return shuffle_out.view(batch_size, channels, out_height)


class PixelShuffle2d(Module):
    """
    2d pixel shuffling
    Shuffles filter dimension into trailing dimensions

    Parameters
    ----------
    scale_kernel : (int,int)
        Factors of how many filters are shuffled

    References
    ----------
    Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A. P., Bishop, R.,
    … Wang, Z. (2016).
    Real-Time Single Image and Video Super-Resolution Using an Efficient
    Sub-Pixel Convolutional Neural Network.
    Retrieved from http://arxiv.org/abs/1609.05158
    """

    def __init__(self, scale_kernel):
        super(PixelShuffle2d, self).__init__()
        self.scale_kernel = scale_kernel

    def forward(self, x, **kwargs):
        batch_size, channels, in_height, in_width = x.size()
        channels /= self.scale_kernel[0] * self.scale_kernel[1]
        channels = int(channels)

        out_height = int(in_height * self.scale_kernel[0])
        out_width = int(in_width * self.scale_kernel[1])

        input_view = x.contiguous().view(
            batch_size, channels, self.scale_kernel[0], self.scale_kernel[1],
            in_height, in_width)

        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
        return shuffle_out.view(batch_size, channels, out_height, out_width)
