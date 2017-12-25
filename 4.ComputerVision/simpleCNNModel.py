import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch import nn


class Conv(nn.Module):
    """
    Custom conv layer
    Assumes the image is squre
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.filters = []
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.filters = Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        # expected size = batch x depth x height x width
        if len(x.size()) != 4:
            raise Exception('Batch should be 4 dimensional')
        batch_size = x.size(0)
        height = x.size(2)
        width = x.size(3)
        new_depth = self.filters.size(0)  # out channels
        new_height = int(((height - self.kernel_size) / self.stride) + 1)
        new_width = int(((width - self.kernel_size) / self.stride) + 1)
        if height != width:
            raise Exception('Only processing square Image')
        # TODO - check whether this converts to cuda tensor if you call .cuda()
        out = Variable(torch.zeros(batch_size, new_depth, new_height, new_width))
        padded_input = F.pad(x, (self.padding,) * 4)
        for nf, f in enumerate(self.filters):
            for h in range(new_height):
                for w in range(new_width):
                    val = padded_input[:, :, h:h + self.kernel_size, w:w + self.kernel_size] * f
                    out[:, nf, h, w] = val.sum()
        return out


class SimpleCNNModel(nn.Module):
    """ A basic CNN model implemented with the the basic building blocks """

    def __init__(self):
        super().__init__()
