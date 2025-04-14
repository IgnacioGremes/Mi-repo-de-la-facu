import numpy as np
import pandas as pd
    
class LogisticRegressionL2:
    def __init__(self, learning_rate=0.01, max_iter=1000, lambda_reg=0.01, tol=1e-4, 
                 cost_fn=1.0, cost_fp=1.0):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.lambda_reg = lambda_reg
        self.tol = tol   
        self.cost_fn = cost_fn        
        self.cost_fp = cost_fp        
        self.weights = None           
        self.bias = None              
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, X, y, weights, bias):
        n_samples = X.shape[0]
        z = np.dot(X, weights) + bias
        y_pred = self.sigmoid(z)
        
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        sample_weights = np.where(y == 1, self.cost_fn, self.cost_fp)
        
        loss = -np.sum(sample_weights * (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))) / n_samples
        
        l2_penalty = (self.lambda_reg / (2 * n_samples)) * np.sum(weights ** 2)
        
        return loss + l2_penalty
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("Target values must be 0 or 1")
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.max_iter):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            error = y_pred - y
            sample_weights = np.where(y == 1, self.cost_fn, self.cost_fp)
            weighted_error = sample_weights * error
            
            grad_weights = (1 / n_samples) * np.dot(X.T, weighted_error) + (self.lambda_reg / n_samples) * self.weights
            grad_bias = (1 / n_samples) * np.sum(weighted_error)
            
            weights_new = self.weights - self.learning_rate * grad_weights
            bias_new = self.bias - self.learning_rate * grad_bias
            
            weight_change = np.linalg.norm(weights_new - self.weights)
            if weight_change < self.tol:
                print(f"Converged after {_ + 1} iterations.")
                break
            
            self.weights = weights_new
            self.bias = bias_new
        
        return self
    
    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def print_coefficients(self):
        print(f"Coefficients: {self.weights}")