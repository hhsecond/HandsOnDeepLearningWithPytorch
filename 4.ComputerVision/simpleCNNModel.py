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
                    val = padded_input[:, :, h:h + self.kernel_size, w:w + self.kernel_size]
                    out[:, nf, h, w] = val.contiguous().view(batch_size, -1) @ f.view(-1)
        return out


class MaxPool(nn.Module):
    """
    Custom max pool layer
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        # expected size = batch x depth x height x width
        if len(x.size()) != 4:
            raise Exception('Batch should be 4 dimensional')
        batch_size = x.size(0)
        depth = x.size(1)
        height = x.size(2)
        width = x.size(3)
        new_height = int(((height - self.kernel_size) / self.kernel_size) + 1)
        new_width = int(((width - self.kernel_size) / self.kernel_size) + 1)
        if height != width:
            raise Exception('Only processing square Image')
        if height % self.kernel_size != 0:
            raise Exception('Keranl cannot be moved completely, change Kernal size')
        out = Variable(torch.zeros(batch_size, depth, new_height, new_width))
        for h in range(new_height):
            for w in range(new_width):
                for d in range(depth):
                    val = x[:, d, h:h + self.kernel_size, w:w + self.kernel_size]
                    out[:, d, h, w] = val.max(2)[0].max(1)[0]
        return out


class SimpleCNNModel(nn.Module):
    """ A basic CNN model implemented with the the basic building blocks """

    def __init__(self):
        super().__init__()
        self.conv1 = Conv(3, 6, 5)
        self.pool = MaxPool(2)
        self.conv2 = Conv(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
