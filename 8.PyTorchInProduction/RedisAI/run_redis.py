import numpy as np
import redis

def binary_encoder(input_size):
    def wrapper(num):
        ret = [int(i) for i in '{0:b}'.format(num)]
        return np.array([[0] * (input_size - len(ret)) + ret], dtype=np.float32)
    return wrapper


def get_readable_output(input_num, prediction):
    input_output_map = {
        0: 'FizBuz',
        1: 'Buz',
        2: 'Fiz'}
    if prediction == 3:
        return input_num
    else:
        return input_output_map[prediction]


r = redis.Redis()
MODEL_PATH = 'fizbuz_model.pt'
with open(MODEL_PATH,'rb') as f:
	model_pt = f.read()
r.execute_command('AI.MODELSET', 'model', 'TORCH', 'CPU', model_pt)
encoder = binary_encoder(10)

while True:
	number = int(input('Enter number, press CTRL+c to exit: ')) + 1
	inputs = encoder(number)

	r.execute_command(
	    'AI.TENSORSET', 'a', 'FLOAT', *inputs.shape, 'BLOB', inputs.tobytes())
	r.execute_command('AI.MODELRUN', 'model', 'INPUTS', 'a', 'OUTPUTS', 'out')
	typ, shape, buf = r.execute_command('AI.TENSORGET', 'out', 'BLOB')
	prediction = np.frombuffer(buf, dtype=np.float32).argmax()
	print(get_readable_output(number, prediction))
