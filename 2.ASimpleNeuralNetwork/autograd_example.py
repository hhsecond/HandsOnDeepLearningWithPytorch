import torch
inputs = torch.FloatTensor([2])
weight = torch.rand(1, requires_grad=True)
bias = torch.ones(1, requires_grad=True)
t = inputs * weight
out = t + bias
out.backward()

weight.grad
bias.grad
