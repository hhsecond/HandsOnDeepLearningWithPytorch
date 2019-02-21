# ~/myp/HOD/8.P/FizBuzTorchScript> python fizbuz.py fizbuz_model.pt 2

import sys
import torch


def main():
    net = torch.jit.load(sys.argv[1])
    temp = [int(i) for i in '{0:b}'.format(int(sys.argv[2]))]
    array = [0] * (10 - len(temp)) + temp
    inputs = torch.Tensor([array])
    print(inputs)  # tensor([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]])
    output = net(inputs)
    print(output)  # tensor([[ -1.8873, -17.1001,  -3.7774,   3.7985]], ...


if __name__ == '__main__':
    main()
