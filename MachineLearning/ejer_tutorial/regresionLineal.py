import numpy as np
import pandas as pd

class RegresionLineal:
    def __init__(self, df, labels_name='y', fit=True):
        assert isinstance(df, pd.DataFrame)
        dfX = df.drop(columns=[labels_name])
        X = np.array(dfX.values).astype(float)
        self.X = np.column_stack((np.ones(X.shape[0]), X))
        self.y = np.array(df[labels_name].values)
        self.features = np.array(['bias'] + list(dfX.columns))
        self.labels_name = labels_name
        self.weights = np.zeros(len(self.features))
        self.weights_trace = [self.weights.copy()]
        if fit:
            self.fit()

    def fit(self, iter_max=1000, alpha=0.01):
        m = len(self.y)
        theta = self.weights.copy()
        X_np = self.X
        y_np = self.y
        
        for _ in range(iter_max):
            # Compute predictions
            h = X_np @ theta  # Matrix multiplication: X_np * theta
            # Compute error
            error = h - y_np
            # Compute gradient
            gradient = (1/m) * (X_np.T @ error)
            # Update parameters
            theta = theta - alpha * gradient
            self.weights_trace.append(theta.copy())
        
        self.weights = theta
        return self.weights

file_path = 'MachineLearning/ejer_tutorial/datasets/datos_oliva.csv'  # Reemplaza con la ruta correcta a tu archivo
df = pd.read_csv(file_path)

# Columnas
ingresos = df['Ingresos']
colesterol = df['colesterol_malo']

# Estandarizacion
df['Ingresos'] = (ingresos - ingresos.min()) /(ingresos.max() - ingresos.min())
# df['colesterol_malo'] = (colesterol - colesterol.mean()) / colesterol.std()

# Regresion
regresion = RegresionLineal(df,labels_name='colesterol_malo')
print(regresion.fit())
# print(regresion.X)
# print(df['Ingresos'])