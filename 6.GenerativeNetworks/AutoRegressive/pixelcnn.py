import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, backends
from torch.utils import data
from torchvision import datasets, transforms, utils
backends.cudnn.benchmark = True

CUDA = torch.cuda.is_available()


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in ('A', 'B')
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


fm = 64
net = nn.Sequential(
    MaskedConv2d('A', 1, fm, 7, 1, 3, bias=False),
    nn.BatchNorm2d(fm),
    nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False),
    nn.BatchNorm2d(fm),
    nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False),
    nn.BatchNorm2d(fm),
    nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False),
    nn.BatchNorm2d(fm),
    nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False),
    nn.BatchNorm2d(fm),
    nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False),
    nn.BatchNorm2d(fm),
    nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False),
    nn.BatchNorm2d(fm),
    nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False),
    nn.BatchNorm2d(fm),
    nn.ReLU(True),
    nn.Conv2d(fm, 256, 1))

device = 'cpu'
if CUDA:
    net.cuda()
    device = 'cuda'

train_data = data.DataLoader(
    datasets.MNIST(
        'data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=128, shuffle=True, num_workers=1)
test_data = data.DataLoader(
    datasets.MNIST(
        'data', train=False, download=True, transform=transforms.ToTensor()),
    batch_size=128, shuffle=False, num_workers=1)
sample = torch.Tensor(144, 1, 28, 28).to(device=device)
optimizer = optim.Adam(net.parameters())
for epoch in range(25):
    err_tr = []
    net.train()
    for i, (input, _) in enumerate(train_data):
        input = input.to(device=device)
        target = (input.data[:, 0] * 255).long()
        loss = F.cross_entropy(net(input), target)
        print(loss.item())
        err_tr.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 5:
            # DO validation
            print('epoch={} - loss={:.7f}'.format(epoch, np.mean(err_tr)))
            sample.fill_(0)
            net.eval()
            with torch.no_grad():
                for i in range(28):
                    for j in range(28):
                        # TODO: put TQDM
                        out = net(sample)
                        # probability assignment for values in the range [0, 255]
                        probs = F.softmax(out[:, :, i, j], dim=1)
                        sample[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.
            utils.save_image(sample, 'sample_{:02d}.png'.format(epoch), nrow=12, padding=0)
