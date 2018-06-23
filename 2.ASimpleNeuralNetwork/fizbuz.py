import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from datautils import get_data, decoder, check_fizbuz

epochs = 500
batches = 64
lr = 0.01
input_size = 10


class FizBuzNet(nn.Module):
    """
    2 layer network for predicting fiz or buz
    param: input_size -> int
    param: output_size -> int
    """

    def __init__(self, input_size, output_size):
        super(FizBuzNet, self).__init__()
        hidden_size = 100
        self.hidden = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, batch):
        hidden = self.hidden(batch)
        activated = F.sigmoid(hidden)
        out = self.out(activated)
        return out


trX, trY, teX, teY = get_data(input_size, limit=1000)
if torch.cuda.is_available():
    xtype = torch.cuda.FloatTensor
    ytype = torch.cuda.LongTensor
else:
    xtype = torch.FloatTensor
    ytype = torch.LongTensor
x = torch.from_numpy(trX).type(xtype)
y = torch.from_numpy(trY).type(ytype)

net = FizBuzNet(input_size, 4)
loss_fn = nn.CrossEntropyLoss()
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


# Test
with torch.no_grad():
    x = torch.from_numpy(teX).type(xtype)
    y = torch.from_numpy(teY).type(ytype)
    hyp = net(x)
    output = loss_fn(hyp, y)
    outli = ['fizbuz', 'buz', 'fiz', 'number']
    for i in range(len(teX)):
        num = decoder(teX[i])
        print(
            'Number: {} -- Actual: {} -- Prediction: {}'.format(
                num, check_fizbuz(num), outli[hyp[i].max(0)[1].item()]))
    print('Test loss: ', output.item() / len(x))
    accuracy = hyp.max(1)[1] == y
    print('accuracy: ', accuracy.sum().item() / len(accuracy))
