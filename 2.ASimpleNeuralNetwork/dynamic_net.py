import random

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


class DynamicNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        for _ in range(random.randint(0, 3)):
            out = F.relu(self.fc2(out))
        return F.relu(self.fc3(out))


batch = 100
in_size = 200
hidden_size = 300
out_size = 10
epochs = 1000
x = torch.randn(batch, in_size)
y = torch.randn(batch, out_size)
net = DynamicNet(in_size, hidden_size, out_size)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)
for epoch in range(epochs):
    targets = net(x)
    loss = loss_fn(targets, y)
    print('Epoch: {:5d} | Loss: {:.5f}'.format(epoch, loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
