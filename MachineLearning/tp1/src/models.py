import numpy as np
import pandas as pd
from preprocessing import normalize

# class LinearRegression:
#     def __init__(self, X, y, normalization='min-max'):
#         """
#         Initialize the linear regression model with normalization.
        
#         Parameters:
#         X (numpy.ndarray or pandas.DataFrame): Feature matrix of shape (n_samples, n_features)
#         y (numpy.ndarray or pandas.Series): Target vector of shape (n_samples,)
#         normalization (str): Type of normalization ('z-score', 'min-max', or None)
#         """
#         # Convert X to NumPy array if it's a pandas DataFrame
#         if hasattr(X, 'values'):
#             X = X.values

#         # Ensure X_new is 2D (even if it's a single feature)
#         if X.ndim == 1:
#             X = X.reshape(-1, 1)

#         # Convert y to NumPy array if it's a pandas Series
#         if hasattr(y, 'values'):
#             y = y.values

#         # Add column of ones for intercept
#         self.X = np.hstack([np.ones((X.shape[0], 1)), X])
#         self.y = y.reshape(-1, 1)  # Ensure y is a column vector
#         self.coef = None
#         self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
#     def fit_pseudo_inverse(self):
#         """
#         Train the model using the pseudo-inverse method.
#         """
#         XTX = np.dot(self.X.T, self.X)
#         XTX_inv = np.linalg.inv(XTX)
#         XTy = np.dot(self.X.T, self.y)
#         self.coef = np.dot(XTX_inv, XTy)
    
#     def fit_gradient_descent(self, learning_rate=0.01, n_iterations=1000):
#         """
#         Train the model using gradient descent.
        
#         Parameters:
#         learning_rate (float): Step size for gradient descent
#         n_iterations (int): Number of iterations
        
#         Returns:
#         numpy.ndarray: The trained coefficients
#         """
#         n_samples, n_features = self.X.shape
#         self.coef = np.zeros((n_features, 1))
        
#         for _ in range(n_iterations):
#             y_pred = np.dot(self.X, self.coef)
#             residuals = y_pred - self.y
#             grad = np.dot(self.X.T, residuals) / n_samples
#             self.coef -= learning_rate * grad
        
#         return self.coef
    
#     def print_coefficients(self, feature_names=None):
#         """
#         Print the coefficients in a clear format.
        
#         Parameters:
#         feature_names (list): Optional list of feature names
#         """
#         if self.coef is None:
#             print("Model has not been trained yet.")
#             return
        
#         if feature_names is not None:
#             self.feature_names = feature_names
        
#         print("Linear Regression Coefficients:")
#         print(f"Intercept (w0): {self.coef[0][0]:.4f}")
#         for i, (coef, name) in enumerate(zip(self.coef[1:], self.feature_names), 1):
#             print(f"{name} (w{i}): {coef[0]:.4f}")
    
#     def predict(self, X_new):
#         """
#         Make predictions on new data.
        
#         Parameters:
#         X_new (numpy.ndarray or pandas.DataFrame): New feature matrix of shape (n_samples_new, n_features)
        
#         Returns:
#         numpy.ndarray: Predicted values of shape (n_samples_new,)
#         """
#         # Convert X_new to NumPy array if it's a pandas DataFrame
#         if hasattr(X_new, 'values'):
#             X_new = X_new.values
        
#         # Ensure X_new is 2D (even if it's a single feature)
#         if X_new.ndim == 1:
#             X_new = X_new.reshape(-1, 1)
        
#         # Add column of ones for intercept
#         X_new = np.hstack([np.ones((X_new.shape[0], 1)), X_new])
#         # Make predictions: y_pred = X_new @ coef
#         y_pred = np.dot(X_new, self.coef)
        
#         return y_pred.flatten()  # Return as 1D array

class LinearRegression:
    def __init__(self, method='gradient_descent', learning_rate=0.01, epochs=1000):
        self.method = method
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None
        self.feature_names = None
    
    def fit(self, X, y, feature_names=None):
        # Convert X to NumPy array if it's a pandas DataFrame
        if hasattr(X, 'values'):
            X = X.values

        # Ensure X_new is 2D (even if it's a single feature)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Convert y to NumPy array if it's a pandas Series
        if hasattr(y, 'values'):
            y = y.values

        X = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        self.feature_names = ['bias'] + (feature_names if feature_names else [f'feature_{i}' for i in range(X.shape[1] - 1)])
        
        if self.method == 'gradient_descent':
            self._fit_gradient_descent(X, y)
        elif self.method == 'pseudo_inverse':
            self._fit_pseudo_inverse(X, y)
        else:
            raise ValueError("Invalid method. Choose 'gradient_descent' or 'pseudo_inverse'")
    
    def _fit_gradient_descent(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)
        for _ in range(self.epochs):
            gradients = (2/m) * X.T @ (X @ self.theta - y)
            self.theta -= self.learning_rate * gradients
    
    def _fit_pseudo_inverse(self, X, y):
        self.theta = np.linalg.pinv(X) @ y  # Compute the normal equation
    
    def predict(self, X):
        # Convert X_new to NumPy array if it's a pandas DataFrame
        if hasattr(X, 'values'):
            X = X.values
        
        # Ensure X_new is 2D (even if it's a single feature)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        return X @ self.theta
    
    def print_coefficients(self):
        if self.theta is None or self.feature_names is None:
            print("Model has not been trained yet.")
        else:
            for name, coef in zip(self.feature_names, self.theta):
                print(f"{name}: {coef}")
    
    



file_path = 'MachineLearning/tp1/data/processed/normalized_train_casas_dev.csv'
df = pd.read_csv(file_path)
X = df.drop(columns=['price'])
Y = df['price']

file_path = 'MachineLearning/tp1/data/processed/cleaned_Amanda.csv'
df_pred = pd.read_csv(file_path)
# df_pred = df_pred.drop(columns=['area_units'])

# # Regresion con solo area

# regresion = LinearRegression(X['area'],Y)
# regresion.fit_pseudo_inverse()

regresion = LinearRegression(method='pseudo_inverse')
regresion.fit(X['area'],Y)

regresion.print_coefficients()

print(regresion.predict(df_pred['area']))

# Regresion

# regresion = LinearRegression(X,Y)
# regresion.fit_pseudo_inverse()

regresion = LinearRegression(method='pseudo_inverse')
regresion.fit(X,Y)

regresion.print_coefficients()

print(regresion.predict(df_pred))

# Regresion with own features

file_path = 'MachineLearning/tp1/data/processed/normalized_own_featured_train_casas_dev.csv'
df = pd.read_csv(file_path)

file_path = 'MachineLearning/tp1/data/processed/featured_Amanda.csv'
df_pred = pd.read_csv(file_path)
# df_pred = df_pred.drop(columns=['area_units'])
X = df.drop(columns=['price'])
Y = df['price']

# regresion = LinearRegression(X,Y)
# regresion.fit_pseudo_inverse()

regresion = LinearRegression(method='pseudo_inverse')
regresion.fit(X,Y)

regresion.print_coefficients()

print(regresion.predict(df_pred))

# Regresion with exponential features

# file_path = 'MachineLearning/tp1/data/processed/exponential_featured_train_casas_dev.csv'
# df = pd.read_csv(file_path)

# file_path = 'MachineLearning/tp1/data/processed/exponential_featured_Amanda.csv'
# df_pred = pd.read_csv(file_path)

# X = df.drop(columns=['price'])
# Y = df['price']

# regresion = LinearRegression(method='pseudo_inverse')
# regresion.fit(X,Y)

# regresion.print_coefficients()

# print(regresion.predict(df_pred))



