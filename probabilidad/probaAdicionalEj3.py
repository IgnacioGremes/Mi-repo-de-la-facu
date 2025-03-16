import numpy as np 
import scipy.stats as scs
import math
import matplotlib.pyplot as plt
np.random.seed(12345)

# a)
N = 200
muestraX1 = scs.norm.rvs(loc = 5, scale = 3, size = N)
muestraX2 = scs.norm.rvs(loc = 8, scale = 4, size = N)
muestraY = muestraX1 + muestraX2

def distrAcumuladaEmprica(muestra,x):
    return sum(x_i <= x for x_i in muestra)/len(muestra)


x = np.arange(min(muestraY),max(muestraY))
plt.figure(figsize=(8,6))
plt.plot(x,distrAcumuladaEmprica(muestraY,x))
plt.plot(x,scs.norm.cdf(x, loc = 13, scale =5))
plt.show()

# b) 
np.random.seed(12345)
CONFIANZA = 0.96
def inf (Xn,var,n,alpha):
    return Xn - scs.norm.isf(alpha/2) * np.sqrt(var/n) 

def sup (Xn,var,n,alpha):
    return Xn + scs.norm.isf(alpha/2) * np.sqrt(var/n) 

XnY = np.mean(muestraY)
alphaY =  1 - CONFIANZA

intervaloDeconfianza96 = [inf(XnY,25,N,alphaY),sup(XnY,25,N,alphaY)]
print(intervaloDeconfianza96)

# c)
def cauntos_datos(varianza,longitud):
    return np.ceil((scs.norm.isf(0.02) * np.sqrt(varianza) * 2 / longitud)**2)