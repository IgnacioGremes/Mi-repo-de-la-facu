import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12345)
N1 = 100
valoresGrafico = np.arange(0,N1)
sample = np.random.exponential(1/0.05,N1)

# Función de probabilidad acumulada empírica
def ecdf(sample, x):
  return sum(x_i <= x for x_i in sample)/len(sample)

def acumuladaExp(lambdaa,valor):
  return 1-np.exp(-lambdaa*valor)

plt.figure(figsize=(8,6))
plt.plot(valoresGrafico,ecdf(sample,valoresGrafico))
plt.plot(valoresGrafico,acumuladaExp(0.05,valoresGrafico))
plt.show()