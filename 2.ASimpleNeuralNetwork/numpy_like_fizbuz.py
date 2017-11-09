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

from datautils import get_data, decoder, check_fizbuz


input_size = 10
output_size = 4
hidden_units = 100
epochs = 1
batches = 64
lr = 0.01


trX, trY, teX, teY = get_data(input_size)
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

x = torch.from_numpy(trX).type(dtype)
x = Variable(x, requires_grad=False)
y = torch.from_numpy(trY).type(dtype)
y = Variable(y, requires_grad=False)

print(x.grad, x.grad_fn, x.data)
# None, None, [torch.FloatTensor of size 900x10]

w1 = torch.randn(input_size, hidden_units).type(dtype)
w1 = Variable(w1, requires_grad=True)
w2 = torch.randn(hidden_units, output_size).type(dtype)
w2 = Variable(w2, requires_grad=True)

print(w1.grad, w1.grad_fn, w1.data)
# None, None, [torch.FloatTensor of size 10x100]

b1 = torch.zeros(1, hidden_units).type(dtype)
b1 = Variable(b1, requires_grad=True)
b2 = torch.zeros(1, output_size).type(dtype)
b2 = Variable(b2, requires_grad=True)

no_of_batches = int(len(trX) / batches)
for epoch in range(epochs):
    for batch in range(no_of_batches):
        start = batch * batches
        end = start + batches
        x_ = x[start:end]
        y_ = y[start:end]

        a2 = x_.matmul(w1)
        a2 = a2.add(b1)

        print(a2.grad, a2.grad_fn, a2.data)
        # None, <AddBackward1 object at 0x7fe4fb786208>, [torch.FloatTensor of size 64x100]

        h2 = a2.sigmoid()

        a3 = h2.matmul(w2)
        a3 = a3.add(b2)
        hyp = a3.sigmoid()

        error = hyp - y_
        output = error.pow(2).sum() / 2.0
        output.backward()

        print(x.grad, x.grad_fn, x.data)
        # None, None, [torch.FloatTensor of size 900x10]
        print(w1.grad, w1.grad_fn, w1.data)
        # [torch.FloatTensor of size 10x100], None, [torch.FloatTensor of size 10x100]
        print(a2.grad, a2.grad_fn, a2.data)
        # None, <AddBackward1 object at 0x7fedcd24f048>, [torch.FloatTensor of size 64x100]

        w1.data -= lr * w1.grad.data
        w2.data -= lr * w2.grad.data
        b1.data -= lr * b1.grad.data
        b2.data -= lr * b2.grad.data
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b1.grad.data.zero_()
        b2.grad.data.zero_()
    if epoch % 10:
        print(epoch, output.data[0])
# traversing the graph using .grad_fn
print(output.grad_fn)
# <DivBackward0 object at 0x7eff00ae3ef0>
print(output.grad_fn.next_functions[0][0])
# <SumBackward0 object at 0x7eff017b4128>
print(output.grad_fn.next_functions[0][0].next_functions[0][0])
# <PowBackward0 object at 0x7eff017b4128>


# test
x = Variable(torch.from_numpy(teX).type(dtype), volatile=True)
y = Variable(torch.from_numpy(teY).type(dtype), volatile=True)

a2 = x.matmul(w1)
a2 = a2.add(b1)
h2 = a2.sigmoid()

a3 = h2.matmul(w2)
a3 = a3.add(b2)
hyp = a3.sigmoid()
error = hyp - y
output = error.pow(2).sum() / 2.
outli = ['fizbuz', 'buz', 'fiz', 'number']
for i in range(len(teX)):
    num = decoder(teX[i])
    print(
        'Number: {} -- Actual: {} -- Prediction: {}'.format(
            num, check_fizbuz(num), outli[hyp[i].data.max(0)[1][0]]))
print('Test loss: ', output.data[0] / len(x))
accuracy = hyp.data.max(1)[1] == y.data.max(1)[1]
print('accuracy: ', accuracy.sum() / len(accuracy))
