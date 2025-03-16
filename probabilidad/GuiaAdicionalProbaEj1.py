import numpy as np
import scipy.stats as scs
import math
import matplotlib.pyplot as plt

np.random.seed(12345)
"""
isf calcual P(X > x) = z, dado el z te halla el x chiquita

"""


# a)
normX =  scs.norm(loc=1500,scale=300)

probabilidadMayor95 = normX.isf(0.05)

print(probabilidadMayor95)

# b)
normY = scs.norm(loc= 21,scale= 5 )

probabilidadDeYMayorA30 = normY.sf(30)

print(probabilidadDeYMayorA30)

# c)
probabilidadDeXMayor2000 = normX.sf(2000)
print(probabilidadDeXMayor2000)
probabilidadDeYMayorA29 = normY.sf(29)
print(probabilidadDeYMayorA29)

probaDeNoAprobarX = 1 - probabilidadDeXMayor2000
probaDeNoAprobarY = 1 - probabilidadDeYMayorA29

probaDeAprobar = 1 - (probaDeNoAprobarX * probaDeNoAprobarY)

print(probaDeAprobar)

# d) 
def quienEstaMejor(puntajeAna,puntajeBeto) -> str:
    rankBeto = normY.sf(puntajeBeto)
    rankAna = normX.sf(puntajeAna)

    if rankBeto < rankAna:
        return "Beto"
    elif rankBeto > rankAna:
        return "Ana"
    return "Iguales"

print(quienEstaMejor(2000,30))