import redis
import numpy as np

r = redis.Redis(host='localhost', port=6379, db=0)

a = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
b = np.asarray([[2, 4], [3, 4]], dtype=np.float32)

with open('addition.py', 'rb') as f:
	script = f.read()

r.execute_command('AI.SCRIPTSET', 'addscript', 'CPU', script)
r.execute_command('AI.TENSORSET', 'a_t', 'FLOAT', '2', '2', 'BLOB', a.tobytes())
r.execute_command('AI.TENSORSET', 'b_t', 'FLOAT', '2', '2', 'BLOB', b.tobytes())
r.execute_command('AI.SCRIPTRUN', 'addscript', 'addition_fn', 'INPUTS', 'a_t', 'b_t', 'OUTPUTS', 'c')
out = r.execute_command('AI.TENSORGET', 'c', 'BLOB')
t = np.frombuffer(out[2], dtype=np.float32).reshape(2, 2)

print(a + b)
"""
[[3. 3.]
 [6. 3.]]
"""


print(t)
"""
[[3. 3.]
 [3. 6.]]
"""

assert (a + b == t).all()