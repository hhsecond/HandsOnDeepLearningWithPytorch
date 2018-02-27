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
y = torch.from_numpy(trY).type(dtype)

print(x.grad, x.grad_fn, x)
# None, None, [torch.FloatTensor of size 900x10]

w1 = torch.randn(input_size, hidden_units, requires_grad=True).type(dtype)
w2 = torch.randn(hidden_units, output_size).type(dtype)

print(w1.grad, w1.grad_fn, w1)
# None, None, [torch.FloatTensor of size 10x100]

b1 = torch.zeros(1, hidden_units, requires_grad=True).type(dtype)
b2 = torch.zeros(1, output_size, requires_grad=True).type(dtype)

no_of_batches = int(len(trX) / batches)
for epoch in range(epochs):
    for batch in range(no_of_batches):
        start = batch * batches
        end = start + batches
        x_ = x[start:end]
        y_ = y[start:end]

        a2 = x_.matmul(w1)
        a2 = a2.add(b1)

        print(a2.grad, a2.grad_fn, a2)
        # None, <AddBackward1 object at 0x7fe4fb786208>, [torch.FloatTensor of size 64x100]

        h2 = a2.sigmoid()

        a3 = h2.matmul(w2)
        a3 = a3.add(b2)
        hyp = a3.sigmoid()

        error = hyp - y_
        output = error.pow(2).sum() / 2.0
        output.backward()

        print(x.grad, x.grad_fn, x)
        # None, None, [torch.FloatTensor of size 900x10]
        print(w1.grad, w1.grad_fn, w1)
        # [torch.FloatTensor of size 10x100], None, [torch.FloatTensor of size 10x100]
        print(a2.grad, a2.grad_fn, a2)
        # None, <AddBackward1 object at 0x7fedcd24f048>, [torch.FloatTensor of size 64x100]

        # Direct manipulation of data outside autograd is not allowed anymore
        # so this code snippet won't work with pytoch version 0.4+
        try:
            w1 -= lr * w1.grad
            w2 -= lr * w2.grad
            b1 -= lr * b1.grad
            b2 -= lr * b2.grad
            w1.grad.zero_()
            w2.grad.zero_()
            b1.grad.zero_()
            b2.grad.zero_()
        except RuntimeError as e:
            raise Exception('Direct manipulation of autograd Variable is not allowed in pytorch \
version 0.4+. Error thrown by pytorch: {}'.format(e))
    if epoch % 10:
        print(epoch, output.item())
# traversing the graph using .grad_fn
print(output.grad_fn)
# <DivBackward0 object at 0x7eff00ae3ef0>
print(output.grad_fn.next_functions[0][0])
# <SumBackward0 object at 0x7eff017b4128>
print(output.grad_fn.next_functions[0][0].next_functions[0][0])
# <PowBackward0 object at 0x7eff017b4128>


# test
with torch.no_grad():
    x = torch.from_numpy(teX).type(dtype)
    y = torch.from_numpy(teY).type(dtype)

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
                num, check_fizbuz(num), outli[hyp[i].max(0)[1].item()]))
    print('Test loss: ', output.item() / len(x))
    accuracy = hyp.max(1)[1] == y.max(1)[1]
    print('accuracy: ', accuracy.sum().item() / len(accuracy))
