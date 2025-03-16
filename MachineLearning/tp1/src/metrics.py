import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

#/home/ig/Desktop/prueba/MachineLearning/tp1/data/raw/casas_dev.csv
file_path = 'MachineLearning/tp1/data/raw/casas_dev.csv'
file_path = 'MachineLearning/tp1/data/processed/cleaned_casas_dev.csv'
df = pd.read_csv(file_path)

plt.figure(figsize=(10, 5))

plt.scatter(df['area'],df['price'])
plt.xlabel('area')
plt.ylabel('price')

plt.show()

plt.figure(figsize=(10, 8))

plt.subplot(2,2,1)
plt.scatter(df['age'],df['price'])
plt.xlabel('age')
plt.ylabel('price')

plt.subplot(2,2,2)
plt.scatter(df['age'],df['area'])
plt.xlabel('age')
plt.ylabel('area')

plt.subplot(2,2,3)
plt.scatter(df['rooms'],df['price'])
plt.xlabel('rooms')
plt.ylabel('price')

plt.subplot(2,2,4)
plt.scatter(df['rooms'],df['area'])
plt.xlabel('rooms')
plt.ylabel('area')
# sb.pairplot(df)
plt.show()

# area vs precio , age vs area y age vs precio (mostrar que se parecen mucho)
# rooms vs area, rooms vs price 