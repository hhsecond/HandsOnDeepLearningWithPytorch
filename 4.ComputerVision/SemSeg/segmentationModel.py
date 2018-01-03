from torch import nn


class SegmentationModel(nn.Module):
    """
    LinkNet for Semantic segmentation. Inspired heavily by
    https://github.com/meetshah1995/pytorch-semseg
    # TODO -> pad = kernal // 2
    """

    def __init__(self):
        super().__init__()
        self.initial_conv = ConvBlock(inp=3, out=64, kernal=7, stride=2, pad=3, bias=False, act=True)
        self.initial_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.encoder1 = EncoderBlock(inp=64, out=64, downsample=False)
        self.encoder2 = EncoderBlock(inp=64, out=128, downsample=True)
        self.encoder3 = EncoderBlock(inp=128, out=256, downsample=True)
        self.encoder4 = EncoderBlock(inp=256, out=512, downsample=True)

        self.decoder4 = DecoderBlock(inp=512, out=256)
        self.decoder3 = DecoderBlock(inp=256, out=128)
        self.decoder2 = DecoderBlock(inp=128, out=64)
        self.decoder1 = DecoderBlock(inp=64, out=64)

        self.final_deconv1 = DeconvBlock(inp=64, out=32, kernal=3, stride=2, pad=1)
        self.final_conv = ConvBlock(inp=32, out=32, kernal=3, stride=1, pad=1, bias=True, act=True)
        self.final_deconv2 = DeconvBlock(inp=32, out=1, kernal=2, stride=2, pad=0)

    def forward(self, x):
        start_conv = self.initial_conv(x)
        start_maxpool = self.initial_maxpool(start_conv)
        print(start_maxpool.size())
        print('################3')
        e1 = self.encoder1(start_maxpool)
        print(e1.size())
        e2 = self.encoder2(e1)
        print(e2.size())
        e3 = self.encoder3(e2)
        print(e3.size())
        e4 = self.encoder4(e3)
        print(e4.size())
        print('################3')

        d4 = self.decoder4(e4, output_size=e3.size()) + e3
        print('<<<<<<<<<<<<<<<<<<')
        print(d4.size())
        d3 = self.decoder3(d4, output_size=e2.size()) + e2
        print(d3.size())
        d2 = self.decoder2(d3, output_size=e1.size()) + e1
        print(d2.size())
        d1 = self.decoder1(d2, output_size=start_maxpool.size())
        print(d1.size())
        print('>>>>>>>>>>>>>>>>>')

        final_deconv1 = self.final_deconv1(d1, output_size=start_conv.size())
        final_conv = self.final_conv(final_deconv1)
        final_deconv2 = self.final_deconv2(final_conv, output_size=x.size())

        return final_deconv2


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
        self.conv1 = ConvBlock(
            inp=inp, out=inp // 4, kernal=1, stride=1, pad=0, bias=True, act=True)
        self.deconv = DeconvBlock(
            inp=inp // 4, out=inp // 4, kernal=3, stride=2, pad=1)
        self.conv2 = ConvBlock(
            inp=inp // 4, out=out, kernal=1, stride=1, pad=0, bias=True, act=True)

    def forward(self, x, output_size):
        conv1 = self.conv1(x)
        deconv = self.deconv(conv1, output_size=output_size)
        conv2 = self.conv2(deconv)
        return conv2


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
        self.conv_transpose = nn.ConvTranspose2d(inp, out, kernal, stride, pad)
        self.batchnorm = nn.BatchNorm2d(out)
        self.relu = nn.ReLU()

    def forward(self, x, output_size):
        print(self, x.size(), output_size)
        convt_out = self.conv_transpose(x, output_size=output_size)
        batchnormout = self.batchnorm(convt_out)
        return self.relu(batchnormout)
