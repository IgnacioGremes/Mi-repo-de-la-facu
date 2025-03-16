import numpy as np 
import scipy.stats as scs
import math
import matplotlib.pyplot as plt
np.random.seed(12345)

# b)

N = 10000
todosLosY =[]
for _ in range(N):
    x = scs.uniform.rvs(loc=0, scale=1 ,size=1)
    y = scs.uniform.rvs(loc=0, scale=x ,size=1)
    todosLosY.append(y)
print(np.mean(todosLosY))

