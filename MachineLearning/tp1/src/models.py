import numpy as np
import pandas as pd
# from preprocessing import normalize

import numpy as np

class LinearRegression:
    def __init__(self, X, y):
        """
        Initialize the linear regression model.
        
        Parameters:
        X (numpy.ndarray or pandas.DataFrame): Feature matrix of shape (n_samples, n_features)
        y (numpy.ndarray or pandas.Series): Target vector of shape (n_samples,)
        """
        # Convert X to NumPy array if it's a pandas DataFrame
        if hasattr(X, 'values'):
            X = X.values
        self.X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add column of ones for intercept
        
        # Convert y to NumPy array if it's a pandas Series
        if hasattr(y, 'values'):
            y = y.values
        self.y = y.reshape(-1, 1)  # Ensure y is a column vector of shape (n_samples, 1)
        self.coef = None
        self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]  # Default feature names
    
    def fit_pseudo_inverse(self):
        """
        Train the model using the pseudo-inverse method.
        """
        XTX = np.dot(self.X.T, self.X)
        XTX_inv = np.linalg.inv(XTX)
        XTy = np.dot(self.X.T, self.y)
        self.coef = np.dot(XTX_inv, XTy)
    
    def fit_gradient_descent(self, learning_rate=0.01, n_iterations=1000):
        """
        Train the model using gradient descent.
        
        Parameters:
        learning_rate (float): Step size for gradient descent
        n_iterations (int): Number of iterations
        
        Returns:
        numpy.ndarray: The trained coefficients
        """
        n_samples, n_features = self.X.shape
        self.coef = np.zeros((n_features, 1))  # Initialize coefficients to zeros
        
        for _ in range(n_iterations):
            # Predicted values: y_pred = X @ coef
            y_pred = np.dot(self.X, self.coef)  # Use np.dot for matrix multiplication
            # Alternatively, you can use self.X @ self.coef if self.X is a NumPy array
            
            # Compute residuals: y_pred - self.y
            residuals = y_pred - self.y  # Both are shape (n_samples, 1)
            
            # Compute gradient: grad = X^T (y_pred - y) / n_samples
            grad = np.dot(self.X.T, residuals) / n_samples  # Shape: (n_features, 1)
            
            # Update coefficients: w = w - learning_rate * grad
            self.coef -= learning_rate * grad
        
        return self.coef
    
    def print_coefficients(self, feature_names=None):
        """
        Print the coefficients in a clear format.
        
        Parameters:
        feature_names (list): Optional list of feature names
        """
        if self.coef is None:
            print("Model has not been trained yet.")
            return
        
        if feature_names is not None:
            self.feature_names = feature_names
        
        print("Linear Regression Coefficients:")
        print(f"Intercept (w0): {self.coef[0][0]:.4f}")
        for i, (coef, name) in enumerate(zip(self.coef[1:], self.feature_names), 1):
            print(f"{name} (w{i}): {coef[0]:.4f}")



file_path = 'MachineLearning/tp1/data/processed/train_cleaned_casas_dev.csv'
df = pd.read_csv(file_path)
print(df)
X = df.drop(columns=['price'])
Y = df['price']

# Regresion
regresion = LinearRegression(X,Y)
regresion.fit_gradient_descent()
regresion.print_coefficients()
