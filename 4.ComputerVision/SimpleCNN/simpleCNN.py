import torch
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms
from simpleCNNModel import SimpleCNNModel


def get_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=100, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    return trainloader, testloader


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = SimpleCNNModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
trainloader, testloader = get_data()

print('Training Started..')
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        check_interval = 2
        if i % check_interval == check_interval - 1:
            print('Iteration: {}, loss: {:.6f}'.format(
                (epoch + 1) * (i + 1), running_loss / check_interval))
            running_loss = 0.0

print('Testing Started..')
correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(images)
    index, value = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (value == labels).sum()

print('Accuracy: {:.5f}'.format(100 * correct / total))
