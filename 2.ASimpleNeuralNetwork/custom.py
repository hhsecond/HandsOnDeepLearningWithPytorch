class Linear(Module):
    def __init__(self, in_features, out_features, bias):
	super(Linear, self).__init__()
	self.in_features = in_features
	self.out_features = out_features
	self.weight = Parameter(torch.Tensor(out_features, in_features))
	self.bias = Parameter(torch.Tensor(out_features))

    def forward(self, input):
	return input.matmul(self.weight.t()) + self.bias



import torch
inputs = torch.FloatTensor([2])
weight = torch.rand(1, requires_grad=True)
bias = torch.ones(1, requires_grad=True)
t = inputs * weight
out = t + bias
out.backward()

weight.grad
bias.grad
