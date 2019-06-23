import torch


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))

    def forward(self, input):
        return input.matmul(self.weight.t()) + self.bias
