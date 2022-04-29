import torch.nn as nn
import torch.nn.functional as F
from msggan.modules import _equalized_conv2d, _equalized_deconv2d, PixelwiseNorm, MinibatchStdDev


class GenInitialBlock(nn.Module):
    """ Module implementing the initial block of the Generator
        Takes in whatever latent size and generates output volume
        of size 4 x 4
    """

    def __init__(self, in_channels, use_eql=True, deconv_size=(2, 8)):
        """
        constructor for the inner class
        :param in_channels: number of input channels to the block
        :param use_eql: whether to use the equalized learning rate
        """
        super().__init__()

        if use_eql:
            self.conv_1 = _equalized_deconv2d(in_channels, in_channels,
                                              deconv_size, bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, in_channels,
                                            (3, 3), pad=1, bias=True)

        else:
            self.conv_1 = nn.ConvTranspose2d(in_channels, in_channels,
                                          deconv_size, bias=True)
            self.conv_2 = nn.Conv2d(in_channels, in_channels, (3, 3),
                                 padding=(1, 1), bias=True)

        # pixel normalization vector:
        self.pixNorm = PixelwiseNorm()

        # leaky_relu:
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the block
        :param x: input to the module
        :return: y => output
        """
        # convert the tensor shape:
        y = x.view(*x.shape, 1, 1)  # add two dummy dimensions for
        # convolution operation

        # perform the forward computations:
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        # apply the pixel normalization:
        y = self.pixNorm(y)

        return y


class GenGeneralConvBlock(nn.Module):
    """ Module implementing a general convolutional block """

    def __init__(self, in_channels, out_channels, use_eql=True):
        """
        constructor for the class
        :param in_channels: number of input channels to the block
        :param out_channels: number of output channels required
        :param use_eql: whether to use the equalized learning rate
        """

        super().__init__()

        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels, out_channels, (3, 3),
                                            pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(out_channels, out_channels, (3, 3),
                                            pad=1, bias=True)

        else:
            self.conv_1 = nn.Conv2d(in_channels, out_channels, (3, 3),
                                 padding=1, bias=True)
            self.conv_2 = nn.Conv2d(out_channels, out_channels, (3, 3),
                                 padding=1, bias=True)

        # pixel_wise feature normalizer:
        self.pixNorm = PixelwiseNorm()

        # leaky_relu:
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the block
        :param x: input
        :return: y => output
        """
        y = F.interpolate(x, scale_factor=2, mode='bilinear')
        y = self.pixNorm(self.lrelu(self.conv_1(y)))
        y = self.pixNorm(self.lrelu(self.conv_2(y)))

        return y


class DisFinalBlock(nn.Module):
    """ Final block for the Discriminator """

    def __init__(self, in_channels, use_eql=True, deconv_size=(2, 8)):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param use_eql: whether to use equalized learning rate
        """

        super().__init__()

        # declare the required modules for forward pass
        self.batch_discriminator = MinibatchStdDev()

        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels + 1, in_channels, (3, 3),
                                            pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, in_channels, deconv_size,
                                            bias=True)

            # final layer emulates the fully connected layer
            self.conv_3 = _equalized_conv2d(in_channels, 1, (1, 1), bias=True)

        else:
            # modules required:
            self.conv_1 = nn.Conv2d(in_channels + 1, in_channels, (3, 3), padding=1, bias=True)
            self.conv_2 = nn.Conv2d(in_channels, in_channels, deconv_size, bias=True)

            # final conv layer emulates a fully connected layer
            self.conv_3 = nn.Conv2d(in_channels, 1, (1, 1), bias=True)

        # leaky_relu:
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the FinalBlock
        :param x: input
        :return: y => output
        """
        # minibatch_std_dev layer
        y = self.batch_discriminator(x)

        # define the computations
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        # fully connected layer
        y = self.conv_3(y)  # This layer has linear activation

        # flatten the output raw discriminator scores
        return y.view(-1)


class DisGeneralConvBlock(nn.Module):
    """ General block in the discriminator  """

    def __init__(self, in_channels, out_channels, use_eql=True):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param use_eql: whether to use equalized learning rate
        """

        super().__init__()

        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels, in_channels, (3, 3),
                                            pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, out_channels, (3, 3),
                                            pad=1, bias=True)
        else:
            # convolutional modules
            self.conv_1 = nn.Conv2d(in_channels, in_channels, (3, 3),
                                 padding=1, bias=True)
            self.conv_2 = nn.Conv2d(in_channels, out_channels, (3, 3),
                                 padding=1, bias=True)

        self.downSampler = nn.AvgPool2d(2)  # downsampler

        # leaky_relu:
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the module
        :param x: input
        :return: y => output
        """
        # define the computations
        y = self.lrelu(self.conv_1(x))
        y = self.lrelu(self.conv_2(y))
        y = self.downSampler(y)

        return y
