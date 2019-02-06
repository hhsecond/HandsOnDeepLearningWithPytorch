import torch
from model import FizBuzNet

input_size = 10
output_size = 4
hidden_size = 100


def get_readable_output(input_num, prediction):
    input_output_map = {
        0: 'FizBuz',
        1: 'Buz',
        2: 'Fiz'}
    if prediction == 3:
        return input_num
    else:
        return input_output_map[prediction]


def binary_encoder():
    def wrapper(num):
        ret = [int(i) for i in '{0:b}'.format(num)]
        return [0] * (input_size - len(ret)) + ret
    return wrapper


net = FizBuzNet(input_size, hidden_size, output_size)
net.load_state_dict(torch.load('fizbuz_model.pth'))
net.eval()
encoder = binary_encoder()


def run(number):
    binary = torch.Tensor([encoder(number)])
    out = net(binary)[0].max(0)[1].item()
    print(number, out)
    return get_readable_output(number, out)
