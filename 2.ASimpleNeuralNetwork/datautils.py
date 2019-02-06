import numpy as np


def binary_encoder(input_size):
    def wrapper(num):
        ret = [int(i) for i in '{0:b}'.format(num)]
        return [0] * (input_size - len(ret)) + ret
    return wrapper


def decoder(array):
    ret = 0
    for i in array:
        ret = ret * 2 + int(i)
    return ret


def training_test_gen(x, y):
    assert len(x) == len(y)
    indices = np.random.permutation(range(len(x)))
    split_size = int(0.9 * len(indices))
    trX = x[indices[:split_size]]
    trY = y[indices[:split_size]]
    teX = x[indices[split_size:]]
    teY = y[indices[split_size:]]
    return trX, trY, teX, teY


def get_pytorch_data(input_size=10, limit=1000):
    x = []
    y = []
    encoder = binary_encoder(input_size)
    for i in range(limit):
        x.append(encoder(i))
        if i % 15 == 0:
            y.append(0)
        elif i % 5 == 0:
            y.append(1)
        elif i % 3 == 0:
            y.append(2)
        else:
            y.append(3)
    return training_test_gen(np.array(x), np.array(y))


def get_numpy_data(input_size=10, limit=1000):
    x = []
    y = []
    encoder = binary_encoder(input_size)
    for i in range(limit):
        x.append(encoder(i))
        if i % 15 == 0:
            y.append([1, 0, 0, 0])
        elif i % 5 == 0:
            y.append([0, 1, 0, 0])
        elif i % 3 == 0:
            y.append([0, 0, 1, 0])
        else:
            y.append([0, 0, 0, 1])
    return training_test_gen(np.array(x), np.array(y))


def check_fizbuz(i):
    if i % 15 == 0:
        return 'fizbuz'
    elif i % 5 == 0:
        return 'buz'
    elif i % 3 == 0:
        return 'fiz'
    else:
        return 'number'
