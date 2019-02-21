import torch
from model import FizBuzNet

input_size = 10
output_size = 4
hidden_size = 100


def binary_encoder():
    def wrapper(num):
        ret = [int(i) for i in '{0:b}'.format(num)]
        return [0] * (input_size - len(ret)) + ret
    return wrapper


net = FizBuzNet(input_size, hidden_size, output_size)
net.load_state_dict(torch.load('fizbuz_model.pth'))
net.eval()
encoder = binary_encoder()


def make_traced_binary(number):
    binary = torch.Tensor([encoder(number)])
    traced = torch.jit.trace(net, binary)
    traced.save('fizbuz_model.pt')


if __name__ == '__main__':
    make_traced_binary(4)
