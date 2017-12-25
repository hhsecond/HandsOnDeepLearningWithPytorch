import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from simpleCNNModel import Conv


###############
# things to write
# Convolution
# 1x1 conv > dim reduction, equivalence with fc
# stride > down sampling
# pool > down sampling (But trends are prefering strides over pooling), max, min, average
###############

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def get_data():
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


myconv = Conv(2, 4, 1)
actualconv = torch.nn.Conv2d(2, 4, 1)
x = Variable(torch.rand(1, 2, 3, 3))
print(myconv(x))
print(actualconv(x))
