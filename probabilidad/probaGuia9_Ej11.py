import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12345)

N = 10000
sample1 = np.random.exponential(1/0.05,N)
sample2 = np.random.exponential(1/0.05,N)
nuevaMuestra = sample1 + sample2
valoresGrafico = np.arange(0,max(nuevaMuestra))
gamma = ss.gamma.pdf(valoresGrafico,2,scale=1/0.05)


max = np.max(nuevaMuestra) 
min = np.min(nuevaMuestra)

plt.figure(figsize=(8,6))
plt.hist(nuevaMuestra,bins= int(np.sqrt(N)),density=True)
plt.plot(valoresGrafico,gamma)
plt.show()