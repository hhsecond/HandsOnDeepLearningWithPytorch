import time
import torch
from torch import nn
import torch.nn.functional as F
from torch import jit
import torch.optim as optim
import numpy as np

input_size = 10
epochs = 2
batches = 64
lr = 0.01


def encoder(input_size):
    def wrapper(num):
        ret = [int(i) for i in '{0:b}'.format(num)]
        return [0] * (input_size - len(ret)) + ret
    return wrapper


def decoder(array):
    ret = 0
    for i in array:
        ret = ret * 2 + int(i)
    return ret


def training_test_gen(x, y):
    assert len(x) == len(y)
    indices = np.random.permutation(range(len(x)))
    split_size = int(0.9 * len(indices))
    trX = x[indices[:split_size]]
    trY = y[indices[:split_size]]
    teX = x[indices[split_size:]]
    teY = y[indices[split_size:]]
    return trX, trY, teX, teY


def get_data(input_size):
    x = []
    y = []
    binary_enc = encoder(input_size)
    for i in range(1000):
        x.append(binary_enc(i))
        if i % 15 == 0:
            y.append([1, 0, 0, 0])
        elif i % 5 == 0:
            y.append([0, 1, 0, 0])
        elif i % 3 == 0:
            y.append([0, 0, 1, 0])
        else:
            y.append([0, 0, 0, 1])
    return training_test_gen(np.array(x), np.array(y))


def check_fizbuz(i):
    if i % 15 == 0:
        return 'fizbuz'
    elif i % 5 == 0:
        return 'buz'
    elif i % 3 == 0:
        return 'fiz'
    else:
        return 'number'


class FizBuzNet(nn.Module):

    def __init__(self, input_size, output_size):
        super(FizBuzNet, self).__init__()
        # A simple heuristic to find the hiddenlayer size
        hidden_size = 100
        self.hidden = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, batch):
        hidden = self.hidden(batch)
        activated = F.sigmoid(hidden)
        out = self.out(activated)
        return F.sigmoid(out)


trX, trY, teX, teY = get_data(input_size)
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor
x = torch.from_numpy(trX).type(dtype)
y = torch.from_numpy(trY).type(dtype)

net = FizBuzNet(input_size, 4)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)
no_of_batches = int(len(trX) / batches)
for epoch in range(epochs):
    for batch in range(no_of_batches):
        optimizer.zero_grad()
        start = batch * batches
        end = start + batches
        x_ = x[start:end]
        y_ = y[start:end]
        hyp = net(x_)
        loss = loss_fn(hyp, y_)
        loss.backward()
        optimizer.step()
