import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import get_data, decoder, check_fizbuz


input_size = 10
epochs = 2000
batches = 64
lr = 0.01


class FizBuzNet(nn.Module):
    """
    2 layer network for predicting fiz or buz
    param: input_size -> int
    param: output_size -> int
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        # A simple heuristic to find the hiddenlayer size
        hidden_size = 100
        self.hidden = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, batch):
        hidden = self.hidden(batch)
        activated = F.sigmoid(hidden)
        out = self.out(activated)
        return F.sigmoid(out)


trX, trY, teX, teY = get_data(input_size)
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor
x = Variable(torch.from_numpy(trX).type(dtype), requires_grad=False)
y = Variable(torch.from_numpy(trY).type(dtype), requires_grad=False)

net = FizBuzNet(input_size, 4)
loss = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=lr)

no_of_batches = int(len(trX) / batches)
for epoch in range(epochs):
    for batch in range(no_of_batches):
        optimizer.zero_grad()
        start = batch * batches
        end = start + batches
        x_ = x[start:end]
        y_ = y[start:end]
        hyp = net(x_)
        output = loss(hyp, y_)
        output.backward()
        optimizer.step()
        print(epoch, output.data[0])


# Test
x = Variable(torch.from_numpy(teX).type(dtype), requires_grad=False)
y = Variable(torch.from_numpy(teY).type(dtype), requires_grad=False)
hyp = net(x)
output = loss(hyp, y)
outli = ['fizbuz', 'buz', 'fiz', 'number']
for i in range(len(teX)):
    num = decoder(teX[i])
    print(
        'Number: {} -- Actual: {} -- Prediction: {}'.format(
            num, check_fizbuz(num), outli[hyp[i].data.max(0)[1][0]]))
print('Test loss: ', loss.data[0])
