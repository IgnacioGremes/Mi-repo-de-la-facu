import numpy as np
import pandas as pd
from preprocessing import normalize
from sklearn.linear_model import LinearRegression

class LinearRegressionn:
    def __init__(self, X, y, normalization='min-max'):
        """
        Initialize the linear regression model with normalization.
        
        Parameters:
        X (numpy.ndarray or pandas.DataFrame): Feature matrix of shape (n_samples, n_features)
        y (numpy.ndarray or pandas.Series): Target vector of shape (n_samples,)
        normalization (str): Type of normalization ('z-score', 'min-max', or None)
        """
        # Convert X to NumPy array if it's a pandas DataFrame
        if hasattr(X, 'values'):
            X = X.values
        # Convert y to NumPy array if it's a pandas Series
        if hasattr(y, 'values'):
            y = y.values

        # Add column of ones for intercept
        self.X = np.hstack([np.ones((X.shape[0], 1)), X])
        self.y = y.reshape(-1, 1)  # Ensure y is a column vector
        self.coef = None
        self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
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
        self.coef = np.zeros((n_features, 1))
        
        for _ in range(n_iterations):
            y_pred = np.dot(self.X, self.coef)
            residuals = y_pred - self.y
            grad = np.dot(self.X.T, residuals) / n_samples
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
    
    def predict(self, X_new):
        """
        Make predictions on new data.
        
        Parameters:
        X_new (numpy.ndarray or pandas.DataFrame): New feature matrix of shape (n_samples_new, n_features)
        
        Returns:
        numpy.ndarray: Predicted values of shape (n_samples_new,)
        """
        # Convert X_new to NumPy array if it's a pandas DataFrame
        if hasattr(X_new, 'values'):
            X_new = X_new.values
        
        # Add column of ones for intercept
        X_new = np.hstack([np.ones((X_new.shape[0], 1)), X_new])
        # Make predictions: y_pred = X_new @ coef
        y_pred = np.dot(X_new, self.coef)
        
        return y_pred.flatten()  # Return as 1D array



file_path = 'MachineLearning/tp1/data/processed/train_cleaned_casas_dev.csv'
df = pd.read_csv(file_path)

file_path = 'MachineLearning/tp1/data/raw/vivienda_Amanda.csv'
df_pred = pd.read_csv(file_path)
df_cleaned = df_pred.drop(columns=['area_units'])
X = df.drop(columns=['price'])
print(X)
Y = df['price']

# Regresion
regresion = LinearRegressionn(X,Y)
regresion.fit_pseudo_inverse()
regresion.print_coefficients()

print(regresion.predict(df_cleaned))

regresos = LinearRegression().fit(X,Y)
print(regresos.predict(df_cleaned))