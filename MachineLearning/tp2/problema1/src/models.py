import numpy as np
import pandas as pd

class LogisticRegressionL2:
    """
    Binary Logistic Regression with L2 regularization.
    
    Parameters:
    learning_rate (float): Step size for gradient descent (default: 0.01)
    max_iter (int): Maximum number of iterations for gradient descent (default: 1000)
    lambda_reg (float): L2 regularization strength (default: 0.01)
    tol (float): Tolerance for convergence (default: 1e-4)
    """
    def __init__(self, learning_rate=0.01, max_iter=1000, lambda_reg=0.01, tol=1e-4):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.lambda_reg = lambda_reg  # L2 regularization parameter
        self.tol = tol                # Convergence tolerance
        self.weights = None           # Model weights
        self.bias = None              # Model bias
    
    def sigmoid(self, z):
        """Compute the sigmoid function."""
        # Clip z to avoid overflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, X, y, weights, bias):
        """Compute the regularized log-loss (cross-entropy + L2 penalty)."""
        n_samples = X.shape[0]
        z = np.dot(X, weights) + bias
        y_pred = self.sigmoid(z)
        
        # Avoid log(0) by clipping predictions
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Cross-entropy loss
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        
        # L2 regularization term (excluding bias)
        l2_penalty = (self.lambda_reg / (2 * n_samples)) * np.sum(weights ** 2)
        
        return loss + l2_penalty
    
    def fit(self, X, y):
        """
        Fit the logistic regression model using gradient descent.
        
        Parameters:
        X (np.ndarray): Training data of shape (n_samples, n_features)
        y (np.ndarray): Target values of shape (n_samples,) with 0 or 1
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.max_iter):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            # Compute gradients
            error = y_pred - y
            grad_weights = (1 / n_samples) * np.dot(X.T, error) + (self.lambda_reg / n_samples) * self.weights
            grad_bias = (1 / n_samples) * np.sum(error)
            
            # Update parameters
            weights_new = self.weights - self.learning_rate * grad_weights
            bias_new = self.bias - self.learning_rate * grad_bias
            
            # Check for convergence
            weight_change = np.linalg.norm(weights_new - self.weights)
            if weight_change < self.tol:
                print(f"Converged after {_ + 1} iterations.")
                break
            
            self.weights = weights_new
            self.bias = bias_new
            
            # Optional: Print loss every 100 iterations
            # if (_ + 1) % 100 == 0:
            #     loss = self.compute_loss(X, y, self.weights, self.bias)
            #     print(f"Iteration {_ + 1}, Loss: {loss:.6f}")
        
        return self
    
    def predict_proba(self, X):
        """
        Predict probability estimates for samples.
        
        Parameters:
        X (np.ndarray): Data of shape (n_samples, n_features)
        
        Returns:
        np.ndarray: Predicted probabilities of shape (n_samples,)
        """
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """
        Predict binary class labels for samples.
        
        Parameters:
        X (np.ndarray): Data of shape (n_samples, n_features)
        threshold (float): Decision threshold (default: 0.5)
        
        Returns:
        np.ndarray: Predicted class labels of shape (n_samples,) with 0 or 1
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def print_coefficients(self):
        print(f"Coefficients: {self.weights}")
    

class LogisticRegressionL2_cost:
    """
    Binary Logistic Regression with L2 regularization and cost-sensitive learning.
    
    Parameters:
    learning_rate (float): Step size for gradient descent (default: 0.01)
    max_iter (int): Maximum number of iterations for gradient descent (default: 1000)
    lambda_reg (float): L2 regularization strength (default: 0.01)
    tol (float): Tolerance for convergence (default: 1e-4)
    cost_fn (float): Cost of false negatives (missing minority class, default: 1.0)
    cost_fp (float): Cost of false positives (incorrectly predicting minority, default: 1.0)
    """
    def __init__(self, learning_rate=0.01, max_iter=1000, lambda_reg=0.01, tol=1e-4, 
                 cost_fn=1.0, cost_fp=1.0):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.lambda_reg = lambda_reg  # L2 regularization parameter
        self.tol = tol                # Convergence tolerance
        self.cost_fn = cost_fn        # Cost of false negatives (minority class)
        self.cost_fp = cost_fp        # Cost of false positives (majority class)
        self.weights = None           # Model weights
        self.bias = None              # Model bias
    
    def sigmoid(self, z):
        """Compute the sigmoid function."""
        # Clip z to avoid overflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, X, y, weights, bias):
        """Compute the cost-sensitive regularized log-loss (cross-entropy + L2 penalty)."""
        n_samples = X.shape[0]
        z = np.dot(X, weights) + bias
        y_pred = self.sigmoid(z)
        
        # Avoid log(0) by clipping predictions
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Assign weights based on class
        sample_weights = np.where(y == 1, self.cost_fn, self.cost_fp)
        
        # Cost-sensitive cross-entropy loss
        loss = -np.sum(sample_weights * (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))) / n_samples
        
        # L2 regularization term (excluding bias)
        l2_penalty = (self.lambda_reg / (2 * n_samples)) * np.sum(weights ** 2)
        
        return loss + l2_penalty
    
    def fit(self, X, y):
        """
        Fit the logistic regression model using gradient descent with cost-sensitive learning.
        
        Parameters:
        X (np.ndarray): Training data of shape (n_samples, n_features)
        y (np.ndarray): Target values of shape (n_samples,) with 0 (majority) or 1 (minority)
        """
        n_samples, n_features = X.shape
        
        # Input validation
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("Target values must be 0 or 1")
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.max_iter):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            # Compute cost-sensitive gradients
            error = y_pred - y
            sample_weights = np.where(y == 1, self.cost_fn, self.cost_fp)
            weighted_error = sample_weights * error
            
            # Gradients with L2 regularization
            grad_weights = (1 / n_samples) * np.dot(X.T, weighted_error) + (self.lambda_reg / n_samples) * self.weights
            grad_bias = (1 / n_samples) * np.sum(weighted_error)
            
            # Update parameters
            weights_new = self.weights - self.learning_rate * grad_weights
            bias_new = self.bias - self.learning_rate * grad_bias
            
            # Check for convergence
            weight_change = np.linalg.norm(weights_new - self.weights)
            if weight_change < self.tol:
                print(f"Converged after {_ + 1} iterations.")
                break
            
            self.weights = weights_new
            self.bias = bias_new
            
            # Optional: Print loss every 100 iterations
            # if (_ + 1) % 100 == 0:
            #     loss = self.compute_loss(X, y, self.weights, self.bias)
            #     print(f"Iteration {_ + 1}, Loss: {loss:.6f}")
        
        return self
    
    def predict_proba(self, X):
        """
        Predict probability estimates for samples.
        
        Parameters:
        X (np.ndarray): Data of shape (n_samples, n_features)
        
        Returns:
        np.ndarray: Predicted probabilities of shape (n_samples,)
        """
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """
        Predict binary class labels for samples.
        
        Parameters:
        X (np.ndarray): Data of shape (n_samples, n_features)
        threshold (float): Decision threshold (default: 0.5)
        
        Returns:
        np.ndarray: Predicted class labels of shape (n_samples,) with 0 or 1
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def print_coefficients(self):
        print(f"Coefficients: {self.weights}")