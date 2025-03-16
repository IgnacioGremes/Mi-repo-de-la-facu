import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12345)

N1 = 100
N2 = 1000
N3 = 10000

sample1 = np.random.normal(7,6,N1)
sample2 = np.random.normal(7,6,N2)
sample3 = np.random.normal(7,6,N3)

esperanza1 = np.mean(sample1)
esperanza2 = np.mean(sample2)
esperanza3 = np.mean(sample3)

varianza1 = np.var(sample1,ddof=1)
varianza2 = np.var(sample2,ddof=1)
varianza3 = np.var(sample3,ddof=1)

print(esperanza1)
print(esperanza2)
print(esperanza3)
print(varianza1)
print(varianza2)
print(varianza3)