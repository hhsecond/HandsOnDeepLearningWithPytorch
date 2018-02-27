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

import time
import torch
from torch import nn
import torch.nn.functional as F
from torch import jit
import torch.optim as optim

from datautils import get_data, decoder, check_fizbuz

input_size = 10
epochs = 500
batches = 64
lr = 0.01


@jit.compile
class FizBuzNet(nn.Module):
    """
    2 layer network for predicting fiz or buz
    param: input_size -> int
    param: output_size -> int
    """

    def __init__(self, input_size, output_size):
        super(FizBuzNet, self).__init__()
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
x = torch.from_numpy(trX).type(dtype)
y = torch.from_numpy(trY).type(dtype)

net = FizBuzNet(input_size, 4)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)
total_time = []
no_of_batches = int(len(trX) / batches)
for epoch in range(epochs):
    for batch in range(no_of_batches):
        optimizer.zero_grad()
        start = batch * batches
        end = start + batches
        x_ = x[start:end]
        y_ = y[start:end]
        start = time.time()
        hyp = net(x_)
        loss = loss_fn(hyp, y_)
        loss.backward()
        total_time.append(time.time() - start)
        optimizer.step()
    if epoch % 10:
        print(epoch, loss.item())
total_sum = sum(total_time)
total_len = len(total_time)
print(total_sum, total_len, total_sum / total_len)
exit()


# Test
with torch.no_grad():
    x = torch.from_numpy(teX).type(dtype)
    y = torch.from_numpy(teY).type(dtype)
    hyp = net(x)
    output = loss_fn(hyp, y)
    outli = ['fizbuz', 'buz', 'fiz', 'number']
    for i in range(len(teX)):
        num = decoder(teX[i])
        print(
            'Number: {} -- Actual: {} -- Prediction: {}'.format(
                num, check_fizbuz(num), outli[hyp[i].max(0)[1].item()]))
    print('Test loss: ', output.item() / len(x))
    accuracy = hyp.max(1)[1] == y.max(1)[1]
    print('accuracy: ', accuracy.sum().item() / len(accuracy))
