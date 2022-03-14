import numpy as np
import math

# MAX_ITER = 10000000
MAX_ITER = 5000000
def f(x):
    # print(x)
    y = [1]*MAX_ITER
    [math.exp(i) for i in y]
def g(x):
    # print(x)
    y = np.ones(MAX_ITER)
    np.exp(y)


from multiprocessing import Pool
def fornorm(f,l):
    for i in l:
        f(i)
import time 
time1 = time.time()
fornorm(g,range(100))
time2 = time.time()
fornorm(f,range(10))
time3 = time.time()

p = Pool(4)
p.map(g,range(100))
time4 = time.time()
p.map(f,range(10))
time5 = time.time()

print(time2 - time1)
print(time3 - time2)
print(time4 - time3)
print(time5 - time4)
