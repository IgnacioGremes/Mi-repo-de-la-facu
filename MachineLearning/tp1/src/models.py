import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, method='gradient_descent', learning_rate=0.01, epochs=1000, L1=0.0, L2=0.0):
        """
        Parameters:
        X: pandas.DataFrame
        y: pandas.Series
        k: Number of folds for cross-validation (default 5)
        method: Training method for LinearRegressionn ('gradient_descent' or 'pseudo_inverse')
        learning_rate: Learning rate for gradient descent
        epochs: Number of iterations for gradient descent
        L1: L1 regularization coefficient
        L2: L2 regularization coefficient
        """
        self.method = method
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.L1 = L1
        self.L2 = L2
        self.coef = None
        self.feature_names = None

    
    def fit(self, X, y):
        # Determine feature names based on the type of X
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = [X.name]
            if hasattr(X, 'values'):
                X = X.values
            if X.ndim == 1:
                X = X.reshape(-1, 1)

        if hasattr(y, 'values'):
            y = y.values

        X = X.values if isinstance(X, pd.DataFrame) else X
        X = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        self.feature_names = ['bias'] + feature_names
        
        if self.method == 'gradient_descent':
            self._fit_gradient_descent(X, y)
        elif self.method == 'pseudo_inverse':
            self._fit_pseudo_inverse(X, y)
        else:
            raise ValueError("Invalid method. Choose 'gradient_descent' or 'pseudo_inverse'")
    
    def _fit_gradient_descent(self, X, y):
        m, n = X.shape
        self.coef = np.zeros(n)
        for _ in range(self.epochs):

            gradients = (2/m) * X.T @ (X @ self.coef - y)
            
            if self.L1 > 0:
                reg_grad_l1 = self.L1 * np.sign(self.coef)
                reg_grad_l1[0] = 0  # Do not regularize the bias term
                gradients += reg_grad_l1
            
            if self.L2 > 0:
                reg_grad_l2 = 2 * self.L2 * self.coef
                reg_grad_l2[0] = 0  # Do not regularize the bias term
                gradients += reg_grad_l2
            
            self.coef -= self.learning_rate * gradients
    
    def _fit_pseudo_inverse(self, X, y):
        if self.L1 > 0:
            raise ValueError("Pseudo-inverse method does not support L1 regularization. Use gradient descent.")
        if self.L2 > 0:
            n_features = X.shape[1]
            I = np.eye(n_features)
            I[0, 0] = 0  # Do not regularize the bias term
            self.coef = np.linalg.inv(X.T @ X + self.L2 * I) @ X.T @ y
        else:
            self.coef = np.linalg.pinv(X) @ y
    
    def predict(self, X):
        if hasattr(X, 'values'):
            X = X.values
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return X @ self.coef
    
    def print_coefficients(self):
        if self.coef is None or self.feature_names is None:
            print("Model has not been trained yet.")
        else:
            print("Model Coefficients:")
            for name, coef in zip(self.feature_names, self.coef):
                print(f"{name}: {coef:.4f}")