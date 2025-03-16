import numpy as np 
import scipy.stats as scs
import math
import matplotlib.pyplot as plt
np.random.seed(12345)

# a)
def intervalo90_varianza(n,mu,sigma):
    muestra = scs.norm.rvs(size=n, loc=mu, scale=sigma)
    varMuestral = np.var(muestra,ddof=1)
    chi2 = scs.chi2(n-1)
    sup = (n-1) * varMuestral / chi2.isf(0.95)
    inf = (n-1) * varMuestral / chi2.isf(0.05)
    return [inf,sup]

# b) 
np.random.seed(12345)

M = 10000
contador = 0
for i in range(M):
    intervalo = intervalo90_varianza(10,2,5)
    if 25 < intervalo[1] and 25 > intervalo[0]:
        contador += 1

print(contador / M)