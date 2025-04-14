import numpy as np
import pandas as pd
from collections import Counter
from math import log2
import random

class LinearDiscriminantAnalysis:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.classes_ = None
        self.means_ = None
        self.priors_ = None
        self.W_ = None
        self.explained_variance_ratio_ = None
        self.covariance_ = None  # For Mahalanobis distance
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        if self.n_components is None:
            self.n_components = min(n_classes - 1, n_features)
        else:
            self.n_components = min(self.n_components, n_classes - 1, n_features)
        
        # Compute class means and priors
        self.means_ = {}
        self.priors_ = {}
        for c in self.classes_:
            X_c = X[y == c]
            self.means_[c] = np.mean(X_c, axis=0)
            self.priors_[c] = X_c.shape[0] / n_samples
        
        # Compute overall mean
        overall_mean = np.mean(X, axis=0)
        
        # Compute within-class scatter matrix (Sw)
        Sw = np.zeros((n_features, n_features))
        for c in self.classes_:
            X_c = X[y == c]
            diff = X_c - self.means_[c]
            Sw += np.dot(diff.T, diff)
        
        # Compute between-class scatter matrix (Sb)
        Sb = np.zeros((n_features, n_features))
        for c in self.classes_:
            diff = (self.means_[c] - overall_mean).reshape(-1, 1)
            Sb += self.priors_[c] * np.dot(diff, diff.T)
        
        # Compute the covariance matrix (for Mahalanobis distance)
        self.covariance_ = Sw / (n_samples - n_classes)
        
        # Solve the generalized eigenvalue problem
        try:
            Sw_inv = np.linalg.inv(Sw)
        except np.linalg.LinAlgError:
            Sw_inv = np.linalg.pinv(Sw)
        
        A = np.dot(Sw_inv, Sb)
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        self.W_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / np.sum(eigenvalues)
        
        return self
    
    def transform(self, X):
        X = np.array(X)
        return np.dot(X, self.W_)
    
    def predict(self, X):
        X_proj = self.transform(X)
        means_proj = {c: np.dot(self.means_[c], self.W_) for c in self.classes_}
        y_pred = []
        for x in X_proj:
            distances = {c: np.linalg.norm(x - means_proj[c]) for c in self.classes_}
            y_pred.append(min(distances, key=distances.get))
        return np.array(y_pred)
    
    def predict_proba(self, X):
        """
        Compute pseudo-probabilities for each class using Mahalanobis distance in the projected space.
        
        Parameters:
        X (np.ndarray): Data of shape (n_samples, n_features)
        
        Returns:
        np.ndarray: Pseudo-probabilities of shape (n_samples, n_classes)
        """
        X = np.array(X)
        X_proj = self.transform(X)
        means_proj = {c: np.dot(self.means_[c], self.W_) for c in self.classes_}
        
        # Compute the covariance matrix in the projected space
        cov_proj = np.dot(self.W_.T, np.dot(self.covariance_, self.W_))
        try:
            cov_inv = np.linalg.inv(cov_proj)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov_proj)
        
        # Compute scores for each class
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for idx, c in enumerate(self.classes_):
            diff = X_proj - means_proj[c]
            # Mahalanobis distance: sqrt((x - mu)^T * Cov^-1 * (x - mu))
            mahalanobis = np.sum(np.dot(diff, cov_inv) * diff, axis=1)
            # Score: -0.5 * Mahalanobis^2 + log(prior)
            scores[:, idx] = -0.5 * mahalanobis + np.log(self.priors_[c])
        
        # Convert scores to pseudo-probabilities using softmax
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))  # Subtract max for numerical stability
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return probs

class LogisticRegressionL2_multi:

    def __init__(self, learning_rate=0.01, max_iter=1000, lambda_reg=0.01, tol=1e-4, class_weights=None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.lambda_reg = lambda_reg
        self.tol = tol
        self.class_weights = class_weights
        self.weights = None
        self.bias = None
        self.classes_ = None
        self.n_classes = None
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def compute_loss(self, X, y, weights, bias):

        n_samples = X.shape[0]

        z = np.dot(X, weights) + bias
        y_pred = self.softmax(z)
        
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        

        y_one_hot = np.zeros((n_samples, self.n_classes))
        for i, label in enumerate(y):
            y_one_hot[i, label - 1] = 1 
        
        if self.class_weights is None:
            loss = -np.mean(np.sum(y_one_hot * np.log(y_pred), axis=1))
        else:
            # Apply class weights for cost-sensitive learning
            sample_weights = np.array([self.class_weights[label] for label in y])
            loss = -np.mean(sample_weights * np.sum(y_one_hot * np.log(y_pred), axis=1))
        
        l2_penalty = (self.lambda_reg / (2 * n_samples)) * np.sum(weights ** 2)
        
        return loss + l2_penalty
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        
        self.weights = np.zeros((n_features, self.n_classes))
        self.bias = np.zeros(self.n_classes)
        
        for iteration in range(self.max_iter):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.softmax(z)
            
            y_one_hot = np.zeros((n_samples, self.n_classes))
            for i, label in enumerate(y):
                y_one_hot[i, label - 1] = 1 
            
            error = y_pred - y_one_hot
            if self.class_weights is not None:
                sample_weights = np.array([self.class_weights[label] for label in y])
                error = error * sample_weights[:, np.newaxis]
            
            grad_weights = (1 / n_samples) * np.dot(X.T, error) + (self.lambda_reg / n_samples) * self.weights
            grad_bias = (1 / n_samples) * np.sum(error, axis=0)
            
            weights_new = self.weights - self.learning_rate * grad_weights
            bias_new = self.bias - self.learning_rate * grad_bias
            
            weight_change = np.linalg.norm(weights_new - self.weights)
            if weight_change < self.tol:
                print(f"Converged after {iteration + 1} iterations.")
                break
            
            self.weights = weights_new
            self.bias = bias_new
            
        return self
    
    def predict_proba(self, X):
        X = np.array(X)
        z = np.dot(X, self.weights) + self.bias
        return self.softmax(z)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
    
    def print_coefficients(self):
        """Print the coefficients for each class."""
        for idx, c in enumerate(self.classes_):
            print(f"Coefficients for class {c}: {self.weights[:, idx]}")
            print(f"Bias for class {c}: {self.bias[idx]}")



class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.classes_ = None 
    
    def entropy(self, y):
        counter = Counter(y)
        total = len(y)
        return -sum((count/total) * log2(count/total) for count in counter.values() if count > 0)
    
    def best_split(self, X, y):
        best_gain = 0
        best_col = None
        best_val = None
        parent_entropy = self.entropy(y)
        
        for col in X.columns:
            values = X[col].unique()
            for val in values:
                left_mask = X[col] <= val
                right_mask = ~left_mask
                
                if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                total = len(y)
                
                gain = parent_entropy
                gain -= (len(y_left)/total) * self.entropy(y_left)
                gain -= (len(y_right)/total) * self.entropy(y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_col = col
                    best_val = val
        
        return best_col, best_val
    
    def build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(set(y)) == 1:
            return Counter(y)
        
        col, val = self.best_split(X, y)
        if col is None:
            return Counter(y)
        
        left_mask = X[col] <= val
        right_mask = ~left_mask
        
        left = self.build_tree(X[left_mask], y[left_mask], depth+1)
        right = self.build_tree(X[right_mask], y[right_mask], depth+1)
        
        return (col, val, left, right)
    
    def fit(self, X, y):
        self.classes_ = sorted(np.unique(y))
        self.tree = self.build_tree(X, y)
    
    def _predict_proba_row(self, row, node):
        if isinstance(node, Counter):
            total = sum(node.values())
            probs = [0.0] * len(self.classes_)
            for cls, count in node.items():
                idx = self.classes_.index(cls)
                probs[idx] = count / total if total > 0 else 0.0
            return probs
        
        col, val, left, right = node
        branch = left if row[col] <= val else right
        return self._predict_proba_row(row, branch)
    
    def predict_proba(self, X):
        return [self._predict_proba_row(row, self.tree) for _, row in X.iterrows()]
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return [self.classes_[np.argmax(p)] for p in probs]

class RandomForestClassifierCustom:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.features = []
        self.classes_ = None 
    
    def _bootstrap(self, X, y):
        n_samples = len(X)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X.iloc[indices], y.iloc[indices]
    
    def _select_features(self, X):
        if self.max_features == 'sqrt':
            n_feats = max(1, int(np.sqrt(X.shape[1])))
        elif self.max_features == 'log2':
            n_feats = max(1, int(np.log2(X.shape[1])))
        elif isinstance(self.max_features, int):
            n_feats = self.max_features
        else:
            n_feats = X.shape[1]
        return random.sample(list(X.columns), n_feats)
    
    def fit(self, X, y):
        self.classes_ = sorted(np.unique(y))
        
        self.trees = []
        self.features = []
        
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap(X, y)
            feat_subset = self._select_features(X_sample)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample[feat_subset], y_sample)
            self.trees.append(tree)
            self.features.append(feat_subset)
    
    def predict_proba(self, X):
        probs = []
        
        for i in range(len(X)):
            prob_sum = [0.0] * len(self.classes_)
            for tree, feats in zip(self.trees, self.features):
                row = X.iloc[[i]][feats]
                tree_probs = tree.predict_proba(row)[0] 
                for idx in range(len(self.classes_)):
                    prob_sum[idx] += tree_probs[idx]
 
            avg_probs = [p / self.n_estimators for p in prob_sum]
            probs.append(avg_probs)
        
        return probs
    
    def predict(self, X):
        probs = self.predict_proba(X)

        return [self.classes_[np.argmax(p)] for p in probs]