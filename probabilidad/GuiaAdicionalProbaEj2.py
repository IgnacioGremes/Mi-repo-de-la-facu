import numpy as np
import scipy.stats as scs
import math
import matplotlib.pyplot as plt

np.random.seed(12345)

# a) 
N = 50

X = scs.uniform(loc=200,scale=100)

esperanzaX = X.median()
print(esperanzaX)
varianzaX = X.var()
print(varianzaX)

esperanzaS50 = N * esperanzaX
print(esperanzaS50)
varianzaS50 = N * varianzaX
print(varianzaS50)

s50 = scs.norm(loc=esperanzaS50, scale= math.sqrt(varianzaS50))

probaDeS50Mayor13 = s50.sf(13000)
print(probaDeS50Mayor13)

#b) 
np.random.seed(12345)
M = 10000
contador = 0
todosLosS50 = []
for i in range(M):
    muestra = scs.uniform.rvs(size=N,loc= 200,scale=100)
    S50 = sum(muestra)
    todosLosS50.append(S50)
    if S50 > 13000:
        contador += 1
print(contador / 10000)

x = np.arange(min(todosLosS50),max(todosLosS50))
plt.figure(figsize=(8,6))
plt.hist(todosLosS50,bins= int(np.sqrt(M)),density=True)
plt.plot(x,s50.pdf(x))
plt.show()
