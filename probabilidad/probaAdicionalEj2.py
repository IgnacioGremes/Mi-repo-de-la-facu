import numpy as np 
import scipy.stats as scs
import math
import matplotlib.pyplot as plt
np.random.seed(12345)

# b)

M = 10000
N =  50
contador = 0
todasS50 = []
for i in range(M):
    muestra = scs.uniform.rvs(size=N,loc=200,scale=100)
    S50 = sum(muestra)
    todasS50.append(S50)
    if(S50 > 13000):
        contador += 1
print(contador / M)

# c) 
np.random.seed(12345)
x = np.arange(min(todasS50),max(todasS50))
normAprox = scs.norm(loc =12500,scale= math.sqrt(125000/3) )


plt.figure(figsize = (8,6))
plt.hist(todasS50, bins=int(math.sqrt(M)) ,density=True)
plt.plot(x,scs.norm.pdf(x,loc =12500,scale= math.sqrt(125000/3)))
plt.show()

# d)
# def cuantoTomates(a,b,p)