import torch
import numpy as np

uninitialized = torch.Tensor(3, 2)
rand_initialized = torch.rand(3, 2)

print(uninitialized)
print(rand_initialized)

# Shape of the tensor
size = rand_initialized.size()
print(size)

# Numpy like API for shape
shape = rand_initialized.shape
print(shape)

# Both are same
print(shape == size)

# indexing through the shape object, same as python tuple
print(shape[0])
print(shape[1])

# Operations
x = torch.ones(3, 2)

# normal operators
y = torch.ones(3, 2) * 2
y = y + 2
z = torch.ones(2, 1)
# x -> 3x2
# y -> 3x2
# z -> 2x1
x_y = x * y  # element wise mul -> 3x2
x_y_z = x_y @ z  # matrix multiplication, (3x2) . (2x1) -> a 3x1 matrix
print(x)
print(y)
print(z)
print(x_y)
print(x_y_z)

# Addition 1 + 2 > 3
z = x + y  # using operators
print(z)

z = x.add(y)  # using pytorch function, torch.add does the same
print(z)

z = x.add_(y)  # in place addition.
print(z)
print(x)  # value after addition

# multiplication
x = torch.rand(2, 3)
y = torch.rand(3, 4)
x.matmul(y)  # tensor of size 2x4

# number of elements in a tensor
x = torch.rand(2, 3)
print(x.numel())

# Slicing, joining, indexing and mutating
# pythonic indexing
x = torch.rand(2, 3, 4)
x_with_2n3_dimension = x[1, :, :]
scalar_x = x[1, 1, 1]  # first value from each dimension

# numpy like slicing
x = torch.rand(2, 3)
print(x[:, 1:])  # skipping first column
print(x[:-1, :])  # skipping last row

# transpose
x = torch.rand(2, 3)
print(x.t())  # size 3x2

# concatenation and stacking
x = torch.rand(2, 3)
concat = torch.cat((x, x))
print(concat)  # Concatenates 2 tensors on default zeroth dimension

x = torch.rand(2, 3)
concat = torch.cat((x, x), dim=1)
print(concat)  # Concatenates 2 tensors on first dimension

x = torch.rand(2, 3)
stacked = torch.stack((x, x), dim=0)
print(stacked)  # concatenated a tensor to new dimension, returns 2x2x3 tensor

# split: you can use chunk as well
x = torch.rand(2, 3)
splitted = x.split(split_size=2, dim=0)  # get 2 tensors of 2 x 2 and 1 x 2 size
print(splitted)

# squeeze and unsqueeze
x = torch.rand(3, 2, 1)  # a tensor of size 3 x 2 x 1
squeezed = x.squeeze()
print(squeezed)  # remove the 1 sized demension

x = torch.rand(3)
with_fake_dimension = x.unsqueeze(0)
print(with_fake_dimension)  # added a fake zeroth dimension

# to numpy
th_tensor = torch.rand(3, 2)
np_tensor = th_tensor.numpy()
print(type(th_tensor), type(np_tensor))

# from numpy
np_tensor = np.random.rand(3, 2)
th_tensor = torch.from_numpy(np_tensor)
print(type(np_tensor), type(th_tensor))

# more fun with th-np conversion
th_arange = torch.arange(1, 4)
np_arange = np.arange(1, 4)
print('would torch-numpy bridge work -> ', th_arange.numpy() == np_arange)

# GPU (CUDA) tensors
x = torch.rand(4, 3)
y = torch.rand(4, 3)
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    z = x + y  # operation executed on GPU
else:
    print('No GPU available')

# this saving and loading method is not the recommended one.
# check here for more info pytorch.org/docs/master/notes/serialization.html
# saving model
x = torch.rand(3, 2)
torch.save(x, 'path')

# load
x = torch.load('path')

# Find more operations here: http://pytorch.org/docs/master/torch.html
