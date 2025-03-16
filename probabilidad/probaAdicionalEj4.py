import numpy as np 
import scipy.stats as scs
import math
import matplotlib.pyplot as plt
np.random.seed(12345)

# a)

def proba_abajo_recta(m):
    intersecRectaConLadoInf = (-2 + m)/m
    return (1-intersecRectaConLadoInf) 

# b)

print(proba_abajo_recta(5))

# c)

N =10000
contador = 0
for i in range(N):
    muestra = scs.uniform.rvs(loc = -1, scale = 2, size = 2)
    if muestra[1] < (5 * muestra[0] -4):
        contador += 1

print(contador / N)