import numpy as np
import redisai
from redisai import Client, BlobTensor, Backend, Device, DType

import redis

def binary_encoder(input_size):
    def wrapper(num):
        ret = [int(i) for i in '{0:b}'.format(num)]
        return np.array([[0] * (input_size - len(ret)) + ret], dtype=np.float32)
    return wrapper

MODEL_PATH = 'fizbuz_model.pt'
number = 3

with open(MODEL_PATH,'rb') as f:
	model_pt = f.read()

encoder = binary_encoder(10)
inputs = encoder(number)

client = Client()
client.modelset('model', Backend.torch, Device.cpu, data=model_pt)
print(inputs.shape)
client.tensorset('a', BlobTensor(DType.float32, inputs.shape, inputs.tobytes()))
client.modelrun('model', input='a', output='out')
final = client.tensorget('out').value
print(final)

# r = redis.Redis()
# r.execute_command('AI.MODELSET', 'model', 'TORCH', 'CPU', model_pt)
# r.execute_command(
#     'AI.TENSORSET', 'a', 'FLOAT', *inputs.shape, 'BLOB', inputs.tobytes())
# r.execute_command('AI.MODELRUN', 'model', 'INPUTS', 'a', 'OUTPUTS', 'out')
# final = r.execute_command('AI.TENSORGET', 'out', 'VALUES')
# print(final)
