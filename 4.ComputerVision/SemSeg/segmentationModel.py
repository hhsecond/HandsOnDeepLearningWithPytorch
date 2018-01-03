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
            ConvBlock(inp=3, out=64, kernal=7, stride=2, pad=3, bias=False, act=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.encoder1 = EncoderBlock(inp=64, out=64, downsample=False)
        self.encoder2 = EncoderBlock(inp=64, out=128, downsample=True)
        self.encoder3 = EncoderBlock(inp=128, out=256, downsample=True)
        self.encoder4 = EncoderBlock(inp=256, out=512, downsample=True)

        self.decoder4 = DecoderBlock(inp=512, out=256)
        self.decoder3 = DecoderBlock(inp=256, out=128)
        self.decoder2 = DecoderBlock(inp=128, out=64)
        self.decoder1 = DecoderBlock(inp=64, out=64)

        self.final_block = nn.Sequential(
            DeconvBlock(inp=64, out=32, kernal=3, stride=2, pad=1),
            ConvBlock(inp=32, out=32, kernal=3, stride=1, pad=1, bias=True, act=True),
            DeconvBlock(inp=32, out=1, kernal=2, stride=2, pad=0))

    def forward(self, x):
        start = self.initial_block(x)
        e1 = self.encoder1(start)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        return e4


class EncoderBlock(nn.Module):
    """ Residucal Block in linknet that does Encoding - layers in ResNet18 """

    def __init__(self, inp, out, downsample):
        """
        Resnet18 has first layer without downsampling.
        The parameter ``downsampling`` decides that
        # TODO - mention about how n - f/s + 1 is handling output size in
        # in downsample
        """
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            self.block1 = nn.Sequential(
                ConvBlock(inp=inp, out=out, kernal=3, stride=2, pad=1, bias=False, act=True),
                ConvBlock(inp=out, out=out, kernal=3, stride=1, pad=1, bias=False, act=False))
            self.residue = ConvBlock(
                inp=inp, out=out, kernal=1, stride=2, pad=0, bias=False, act=False)
        else:
            self.block1 = nn.Sequential(
                ConvBlock(inp=inp, out=out, kernal=3, stride=1, pad=1, bias=False, act=True),
                ConvBlock(inp=out, out=out, kernal=3, stride=1, pad=1, bias=False, act=False))
        self.block2 = nn.Sequential(
            ConvBlock(inp=out, out=out, kernal=3, stride=1, pad=1, bias=False, act=True),
            ConvBlock(inp=out, out=out, kernal=3, stride=1, pad=1, bias=False, act=False))

    def forward(self, x):
        out1 = self.block1(x)
        if self.downsample:
            residue = self.residue(x)
            out2 = self.block2(out1 + residue)
        else:
            out2 = self.block2(out1 + x)
        return out2 + out1


class DecoderBlock(nn.Module):
    """ Residucal Block in linknet that does Encoding """

    def __init__(self, inp, out):
        super().__init__()

    def forward(self, x):
        pass


class ConvBlock(nn.Module):
    """ LinkNet uses initial block with conv -> batchnorm -> relu """

    def __init__(self, inp, out, kernal, stride, pad, bias, act):
        super().__init__()
        if act:
            self.conv_block = nn.Sequential(
                nn.Conv2d(inp, out, kernal, stride, pad, bias=bias),
                nn.BatchNorm2d(num_features=out),
                nn.ReLU())
        else:
            self.conv_block = nn.Sequential(
                nn.Conv2d(inp, out, kernal, stride, pad, bias=bias),
                nn.BatchNorm2d(num_features=out))

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
