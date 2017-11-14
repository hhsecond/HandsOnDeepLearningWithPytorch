##########################################################################
##########################################################################
# Author: Sherin Thomas
# PyTorch Version 0.2
# Program for predicting the next number whether its a
# fiz or buz or fizbuz
# The original post written the idea of fizbuz in neural network:
#       http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/
#
#
#
# incase you haven't played this game, you had the worst childhood,
# but don't worry, here are the rules.
#
# number divisible by 3 is fiz
# number divisible by 5 is buz
# number divisible by both 3 and 5 is fizbuz
# other numbers should be considered as that number itself
###########################################################################
###########################################################################

import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from datautils import get_data, decoder, check_fizbuz

input_size = 10
epochs = 500
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
optimizer = optim.Adam(net.parameters(), lr=lr)

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
    if epoch % 10:
        print(epoch, output.data[0])


# Test
x = Variable(torch.from_numpy(teX).type(dtype), volatile=True)
y = Variable(torch.from_numpy(teY).type(dtype), volatile=True)
hyp = net(x)
output = loss(hyp, y)
outli = ['fizbuz', 'buz', 'fiz', 'number']
for i in range(len(teX)):
    num = decoder(teX[i])
    print(
        'Number: {} -- Actual: {} -- Prediction: {}'.format(
            num, check_fizbuz(num), outli[hyp[i].data.max(0)[1][0]]))
print('Test loss: ', output.data[0] / len(x))
accuracy = hyp.data.max(1)[1] == y.data.max(1)[1]
print('accuracy: ', accuracy.sum() / len(accuracy))
