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

import torch as th
from torch.autograd import Variable

from dataset import get_data, decoder, check_fizbuz


input_size = 10
epochs = 2000
batches = 64
lr = 0.01


trX, trY, teX, teY = get_data(input_size)
if th.cuda.is_available():
    dtype = th.cuda.FloatTensor
else:
    dtype = th.FloatTensor
x = Variable(th.from_numpy(trX).type(dtype), requires_grad=False)
y = Variable(th.from_numpy(trY).type(dtype), requires_grad=False)


w1 = Variable(th.randn(10, 100).type(dtype), requires_grad=True)
w2 = Variable(th.randn(100, 4).type(dtype), requires_grad=True)

b1 = Variable(th.zeros(1, 100).type(dtype), requires_grad=True)
b2 = Variable(th.zeros(1, 4).type(dtype), requires_grad=True)

no_of_batches = int(len(trX) / batches)
for epoch in range(epochs):
    for batch in range(no_of_batches):
        start = batch * batches
        end = start + batches
        x_ = x[start:end]
        y_ = y[start:end]

        a2 = x_.matmul(w1)
        a2 = a2.add(b1)
        h2 = a2.sigmoid()

        a3 = h2.matmul(w2)
        a3 = a3.add(b2)
        hyp = a3.sigmoid()

        error = hyp - y_
        loss = error.pow(2).sum() / 2.0
        loss.backward()

        w1.data -= lr * w1.grad.data
        w2.data -= lr * w2.grad.data
        b1.data -= lr * b1.grad.data
        b2.data -= lr * b2.grad.data
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b1.grad.data.zero_()
        b2.grad.data.zero_()
    print(epoch, loss.data[0])


# test
x = Variable(th.from_numpy(teX).type(dtype), requires_grad=False)
y = Variable(th.from_numpy(teY).type(dtype), requires_grad=False)

a2 = x.matmul(w1)
a2 = a2.add(b1)
h2 = a2.sigmoid()

a3 = h2.matmul(w2)
a3 = a3.add(b2)
hyp = a3.sigmoid()
error = hyp - y
loss = error.pow(2).sum() / 2.
outli = ['fizbuz', 'buz', 'fiz', 'number']
for i in range(len(teX)):
    num = decoder(teX[i])
    print(
        'Number: {} -- Actual: {} -- Prediction: {}'.format(
            num, check_fizbuz(num), outli[hyp[i].data.max(0)[1][0]]))
print('Test loss: ', loss.data[0])
