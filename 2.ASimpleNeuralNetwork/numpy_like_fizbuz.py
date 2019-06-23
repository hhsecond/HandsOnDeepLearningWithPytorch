import torch

from datautils import get_numpy_data, decoder, check_fizbuz


epochs = 5
batches = 64
lr = 0.01
input_size = 10
output_size = 4
hidden_size = 100


trX, trY, teX, teY = get_numpy_data(input_size)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float64


x = torch.from_numpy(trX).to(device=device, dtype=dtype)
y = torch.from_numpy(trY).to(device=device, dtype=dtype)

print(x.grad, x.grad_fn, x)
# None None tensor([[...]])


w1 = torch.randn(input_size, hidden_size, requires_grad=True, device=device, dtype=dtype)
w2 = torch.randn(hidden_size, output_size, requires_grad=True, device=device, dtype=dtype)

print(w1.grad, w1.grad_fn, w1)
# None None tensor([[...]])

b1 = torch.zeros(1, hidden_size, requires_grad=True, device=device, dtype=dtype)
b2 = torch.zeros(1, output_size, requires_grad=True, device=device, dtype=dtype)

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
        # None <AddBackward0 object at 0x7f5f3b9253c8> tensor([[...]])

        h2 = a2.sigmoid()

        a3 = h2.matmul(w2)
        a3 = a3.add(b2)
        hyp = a3.sigmoid()

        error = hyp - y_
        output = error.pow(2).sum() / 2.0
        output.backward()

        print(x.grad, x.grad_fn, x)
        # None None tensor([[...]])
        print(w1.grad, w1.grad_fn, w1)
        # tensor([[...]], None, tensor([[...]]
        print(a2.grad, a2.grad_fn, a2)
        # None <AddBackward0 object at 0x7f5f3d42c780> tensor([[...]])

        # Direct manipulation of data outside autograd is not allowed
        # when grad flag is True
        with torch.no_grad():
            w1 -= lr * w1.grad
            w2 -= lr * w2.grad
            b1 -= lr * b1.grad
            b2 -= lr * b2.grad
        # Making gradients zero. This is essential otherwise, gradient
        # from next iteration accumulates
        w1.grad.zero_()
        w2.grad.zero_()
        b1.grad.zero_()
        b2.grad.zero_()
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
    x = torch.from_numpy(teX).to(device=device, dtype=dtype)
    y = torch.from_numpy(teY).to(device=device, dtype=dtype)

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
