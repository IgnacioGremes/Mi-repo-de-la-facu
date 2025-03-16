import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

file_path = '/home/ig/Desktop/prueba/problemasFia/recetas_muffins_cupcakes_scones.csv'

df = pd.read_csv(file_path)

y = df['Type']
x = df.drop(columns=['Type'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

#%%
# Logistic Regression
model = LogisticRegression()

model.fit(x_train, y_train)

print('Logistic Regression:',model.score(x_test, y_test))
# %%
# Plot milk vs sugar

sns.scatterplot(x='Butter', y='Sugar', hue='Type', data=df)

# %%
# knn

model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)