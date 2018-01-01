from torch import nn


class SegmentationModel(nn.Module):
    """
    LinkNet for Semantic segmentation. Inspired heavily by
    https://github.com/meetshah1995/pytorch-semseg
    # TODO -> pad = kernal // 2
    """

    def __init__(self):
        super().__init__()
        self.initial_block = nn.Sequential(
            ConvBlock(inp=3, out=64, kernal=7, stride=2, pad=3, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.encoder1 = EncoderBlock()
        self.encoder2 = EncoderBlock()
        self.encoder3 = EncoderBlock()
        self.encoder4 = EncoderBlock()

        self.decoder4 = DecoderBlock()
        self.decoder3 = DecoderBlock()
        self.decoder2 = DecoderBlock()
        self.decoder1 = DecoderBlock()

        self.final_block = nn.Sequential(
            DeconvBlock(inp=64, out=32, kernal=3, stride=2, pad=1),
            ConvBlock(inp=32, out=32, kernal=3, stride=1, pad=1, bias=True),
            DeconvBlock(inp=32, out=1, kernal=2, stride=2, pad=0))

    def forward(self, x):
        start = self.initial_block(x)
        return start


class EncoderBlock(nn.Module):
    """ Residucal Block in linknet that does Encoding """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class DecoderBlock(nn.Module):
    """ Residucal Block in linknet that does Encoding """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class ConvBlock(nn.Module):
    """ LinkNet uses initial block with conv -> batchnorm -> relu """

    def __init__(self, inp, out, kernal, stride, pad, bias):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(inp, out, kernal, stride, pad, bias=bias),
            nn.BatchNorm2d(num_features=out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv_block(x)


class DeconvBlock(nn.Module):
    """ LinkNet uses Deconv block with transposeconv -> batchnorm -> relu """

    def __init__(self, inp, out, kernal, stride, pad):
        super().__init__()
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(inp, out, kernal, stride, pad),
            nn.BatchNorm2d(out),
            nn.ReLU())

    def forward(self, x):
        return self.deconv_block(x)
