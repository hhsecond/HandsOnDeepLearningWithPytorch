# copied from Apazke's original tutorial @ https://pytorch.org/tutorials/advanced/numpy_extensions_tutorial.html
from scipy.signal import convolve2d, correlate2d
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch
from torch.autograd import Function
from numpy.fft import rfft2, irfft2


class BadFFTFunction(Function):

    def forward(self, input):
        numpy_input = input.detach().numpy()
        result = abs(rfft2(numpy_input))
        return input.new(result)

    def backward(self, grad_output):
        numpy_go = grad_output.numpy()
        result = irfft2(numpy_go)
        return grad_output.new(result)


def incorrect_fft(input):
    return BadFFTFunction()(input)


input = torch.randn(8, 8, requires_grad=True)
result = incorrect_fft(input)
print(result)
result.backward(torch.randn(result.size()))
print(input)


class ScipyConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, filter):
        input, filter = input.detach(), filter.detach()  # detach so we can cast to NumPy
        result = correlate2d(input.numpy(), filter.detach().numpy(), mode='valid')
        ctx.save_for_backward(input, filter)
        return input.new(result)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.detach()
        input, filter = ctx.saved_tensors
        grad_input = convolve2d(grad_output.numpy(), filter.t().numpy(), mode='full')
        grad_filter = convolve2d(input.numpy(), grad_output.numpy(), mode='valid')

        return grad_output.new_tensor(grad_input), grad_output.new_tensor(grad_filter)


class ScipyConv2d(Module):

    def __init__(self, kh, kw):
        super(ScipyConv2d, self).__init__()
        self.filter = Parameter(torch.randn(kh, kw))

    def forward(self, input):
        return ScipyConv2dFunction.apply(input, self.filter)


module = ScipyConv2d(3, 3)
print(list(module.parameters()))
input = torch.randn(10, 10, requires_grad=True)
output = module(input)
print(output)
output.backward(torch.randn(8, 8))
print(input.grad)
