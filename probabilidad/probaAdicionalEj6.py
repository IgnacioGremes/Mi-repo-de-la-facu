import numpy as np 
import scipy.stats as scs
import math
import matplotlib.pyplot as plt
np.random.seed(12345)

M = 10000
N = 20
valores =[]
chi2 = scs.chi2(N-1)

for i in range(M):
    muestra = scs.norm.rvs(loc= 2, scale= np.sqrt(3), size=N)
    varMuestral = np.var(muestra,ddof=1)
    valores.append((N-1)/ 3 * varMuestral) 

x = np.arange(min(valores),max(valores))

plt.figure(figsize=(8,6))
plt.hist(valores, bins= int(np.sqrt(M)),density=True)
plt.plot(x,chi2.pdf(x))
plt.show()
