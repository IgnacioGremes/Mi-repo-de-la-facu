import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# import csv

path = "recetas_muffins_cupcakes_scones.csv"
df = pd.read_csv(path)

df.head()

y = df["Type"]
X = df.drop(columns=["Type"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %% Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
print("Logistic Regression:", model.score(X_test, y_test))

# plot milk vs sugar
sns.scatterplot(x="Butter", y="Sugar", hue="Type", data=df)


# %% plot model decision boundary using meshgrid

def plot_decision_boundary(model, X, y):
    h = .2  # step size in the mesh
    x_min, x_max = X["Butter"].min() - 1, X["Butter"].max() + 1
    y_min, y_max = X["Sugar"].min() - 1, X["Sugar"].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # map clases to numbers Muffin Cupcake Scone
    cmap = {"Muffin": 0, "Cupcake": 1, "Scone": 2}
    Z = np.vectorize(cmap.get)(Z)

    # plot mshgrid points
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.show()


#plot_decision_boundary(model, X, y)

# %% KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
print("KNN:", model.score(X_test, y_test))

#plot_decision_boundary(model, X, y)
# %% Decision Tree

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print("Decision Tree:", model.score(X_test, y_test))

#plot_decision_boundary(model, X, y)

# plot tree
from sklearn.tree import plot_tree

plt.figure(figsize=(10, 10))
plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)

# %% Random Forest

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Random Forest:", model.score(X_test, y_test))

#plot_decision_boundary(model, X, y)
# %% gradient boosting

from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model.fit(X_train, y_train)

print("Gradient Boosting:", model.score(X_test, y_test))

#plot_decision_boundary(model, X, y)
# %%
