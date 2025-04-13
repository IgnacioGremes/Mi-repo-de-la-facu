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

# class DecisionTree:
#     """
#     Decision Tree classifier using entropy as the splitting criterion.
#     """
#     def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.max_features = max_features
#         self.tree_ = None
    
#     def entropy(self, y):
#         """Compute entropy of a set of labels."""
#         classes, counts = np.unique(y, return_counts=True)
#         probs = counts / len(y)
#         return -np.sum(probs * np.log2(probs + 1e-10))  # Add small value to avoid log(0)
    
#     def information_gain(self, y, y_left, y_right):
#         """Compute information gain from a split."""
#         parent_entropy = self.entropy(y)
#         n = len(y)
#         n_left, n_right = len(y_left), len(y_right)
#         if n_left == 0 or n_right == 0:
#             return 0
#         child_entropy = (n_left / n) * self.entropy(y_left) + (n_right / n) * self.entropy(y_right)
#         return parent_entropy - child_entropy
    
#     def find_best_split(self, X, y, feature_indices):
#         """Find the best split for a node."""
#         best_gain = -float('inf')
#         best_feature = None
#         best_threshold = None
#         best_left_mask = None
        
#         for feature in feature_indices:
#             values = X[:, feature]
#             thresholds = np.unique(values)
            
#             for threshold in thresholds:
#                 left_mask = values <= threshold
#                 right_mask = ~left_mask
                
#                 if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
#                     continue
                
#                 gain = self.information_gain(y, y[left_mask], y[right_mask])
#                 if gain > best_gain:
#                     best_gain = gain
#                     best_feature = feature
#                     best_threshold = threshold
#                     best_left_mask = left_mask
        
#         return best_feature, best_threshold, best_left_mask, best_gain
    
#     def fit(self, X, y, depth=0):
#         """Fit the decision tree."""
#         X = np.array(X)
#         y = np.array(y)
#         n_samples, n_features = X.shape
        
#         # Stopping criteria
#         if (self.max_depth is not None and depth >= self.max_depth) or \
#            n_samples < self.min_samples_split or \
#            len(np.unique(y)) == 1:
#             return {'leaf': True, 'class': np.bincount(y).argmax()}
        
#         # Select random subset of features
#         n_features_to_consider = self.max_features if self.max_features is not None else n_features
#         feature_indices = np.random.choice(n_features, n_features_to_consider, replace=False)
        
#         # Find the best split
#         best_feature, best_threshold, best_left_mask, best_gain = self.find_best_split(X, y, feature_indices)
        
#         if best_feature is None or best_gain <= 0:
#             return {'leaf': True, 'class': np.bincount(y).argmax()}
        
#         # Split the data
#         X_left, y_left = X[best_left_mask], y[best_left_mask]
#         X_right, y_right = X[~best_left_mask], y[~best_left_mask]
        
#         # Recursively build the tree
#         left_subtree = self.fit(X_left, y_left, depth + 1)
#         right_subtree = self.fit(X_right, y_right, depth + 1)
        
#         return {
#             'leaf': False,
#             'feature': best_feature,
#             'threshold': best_threshold,
#             'left': left_subtree,
#             'right': right_subtree
#         }
    
#     def predict_one(self, x, node):
#         """Predict the class for a single sample."""
#         if node['leaf']:
#             return node['class']
        
#         if x[node['feature']] <= node['threshold']:
#             return self.predict_one(x, node['left'])
#         else:
#             return self.predict_one(x, node['right'])
    
#     def predict(self, X, node):
#         """Predict classes for all samples."""
#         X = np.array(X)
#         return np.array([self.predict_one(x, node) for x in X])
    
#     def predict_proba_one(self, x, node):
#         """Predict class probabilities for a single sample (based on leaf distribution)."""
#         if node['leaf']:
#             class_counts = np.bincount([node['class']], minlength=len(np.unique([node['class']])))
#             return class_counts / np.sum(class_counts)
        
#         if x[node['feature']] <= node['threshold']:
#             return self.predict_proba_one(x, node['left'])
#         else:
#             return self.predict_proba_one(x, node['right'])
    
#     def predict_proba(self, X, node):
#         """Predict class probabilities for all samples."""
#         X = np.array(X)
#         return np.array([self.predict_proba_one(x, node) for x in X])

# class RandomForest:
#     """
#     Random Forest classifier using entropy as the splitting criterion.
    
#     Parameters:
#     n_trees (int): Number of trees in the forest (default: 100)
#     max_depth (int, optional): Maximum depth of each tree (default: None)
#     min_samples_split (int): Minimum number of samples required to split a node (default: 2)
#     max_features (int or str, optional): Number of features to consider at each split
#         (default: 'sqrt', i.e., sqrt(n_features))
#     random_state (int, optional): Random seed for reproducibility (default: None)
#     """
#     def __init__(self, n_trees=100, max_depth=None, min_samples_split=2, 
#                  max_features='sqrt', random_state=None):
#         self.n_trees = n_trees
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.max_features = max_features
#         self.random_state = random_state
#         self.trees_ = []
#         self.classes_ = None
    
#     def fit(self, X, y):
#         """Fit the random forest."""
#         X = np.array(X)
#         y = np.array(y)
#         n_samples, n_features = X.shape
        
#         # Set random seed
#         if self.random_state is not None:
#             np.random.seed(self.random_state)
        
#         # Determine max_features
#         if self.max_features == 'sqrt':
#             self.max_features = int(np.sqrt(n_features))
#         elif self.max_features == 'log2':
#             self.max_features = int(np.log2(n_features))
#         elif isinstance(self.max_features, int):
#             self.max_features = min(self.max_features, n_features)
#         else:
#             self.max_features = n_features
        
#         # Get unique classes
#         self.classes_ = np.unique(y)
        
#         # Build each tree
#         self.trees_ = []
#         for _ in range(self.n_trees):
#             # Bootstrap sample
#             indices = np.random.choice(n_samples, n_samples, replace=True)
#             X_boot = X[indices]
#             y_boot = y[indices]
            
#             # Create and fit a decision tree
#             tree = DecisionTree(
#                 max_depth=self.max_depth,
#                 min_samples_split=self.min_samples_split,
#                 max_features=self.max_features
#             )
#             tree.tree_ = tree.fit(X_boot, y_boot)
#             self.trees_.append(tree)
        
#         return self
    
#     def predict(self, X):
#         """Predict class labels for samples in X."""
#         X = np.array(X)
        
#         # Get predictions from all trees
#         predictions = np.array([tree.predict(X, tree.tree_) for tree in self.trees_])
        
#         # Majority voting
#         y_pred = []
#         for i in range(X.shape[0]):
#             votes = predictions[:, i]
#             y_pred.append(np.bincount(votes).argmax())
        
#         return np.array(y_pred)
    
#     def predict_proba(self, X):
#         """Predict class probabilities for samples in X."""
#         X = np.array(X)
        
#         # Get probability estimates from all trees
#         n_classes = len(self.classes_)
#         proba_sum = np.zeros((X.shape[0], n_classes))
        
#         for tree in self.trees_:
#             proba = tree.predict_proba(X, tree.tree_)
#             # Adjust proba to match the number of classes
#             proba_adjusted = np.zeros((X.shape[0], n_classes))
#             for idx, c in enumerate(tree.predict(X, tree.tree_)):
#                 proba_adjusted[idx, c-1] = proba[idx, 0] if proba.shape[1] == 1 else proba[idx, c-1]
#             proba_sum += proba_adjusted
        
#         # Average probabilities across trees
#         proba = proba_sum / self.n_trees
        
#         return proba
    













# class DecisionTree:
#     def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.max_features = max_features
#         self.tree_ = None
#         self.n_classes = None  # To store the total number of classes
    
#     def entropy(self, y):
#         classes, counts = np.unique(y, return_counts=True)
#         probs = counts / len(y)
#         return -np.sum(probs * np.log2(probs + 1e-10))
    
#     def information_gain(self, y, y_left, y_right):
#         parent_entropy = self.entropy(y)
#         n = len(y)
#         n_left, n_right = len(y_left), len(y_right)
#         if n_left == 0 or n_right == 0:
#             return 0
#         child_entropy = (n_left / n) * self.entropy(y_left) + (n_right / n) * self.entropy(y_right)
#         return parent_entropy - child_entropy
    
#     def find_best_split(self, X, y, feature_indices):
#         best_gain = -float('inf')
#         best_feature = None
#         best_threshold = None
#         best_left_mask = None
        
#         for feature in feature_indices:
#             values = X[:, feature]
#             thresholds = np.unique(values)
            
#             for threshold in thresholds:
#                 left_mask = values <= threshold
#                 right_mask = ~left_mask
                
#                 if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
#                     continue
                
#                 gain = self.information_gain(y, y[left_mask], y[right_mask])
#                 if gain > best_gain:
#                     best_gain = gain
#                     best_feature = feature
#                     best_threshold = threshold
#                     best_left_mask = left_mask
        
#         return best_feature, best_threshold, best_left_mask, best_gain
    
#     def fit(self, X, y, depth=0):
#         X = np.array(X)
#         y = np.array(y)
#         n_samples, n_features = X.shape
        
#         # Store the total number of classes
#         self.n_classes = len(np.unique(y))
        
#         # Stopping criteria
#         if (self.max_depth is not None and depth >= self.max_depth) or \
#            n_samples < self.min_samples_split or \
#            len(np.unique(y)) == 1:
#             return {'leaf': True, 'class': np.bincount(y).argmax()}
        
#         # Select random subset of features
#         n_features_to_consider = self.max_features if self.max_features is not None else n_features
#         feature_indices = np.random.choice(n_features, n_features_to_consider, replace=False)
        
#         # Find the best split
#         best_feature, best_threshold, best_left_mask, best_gain = self.find_best_split(X, y, feature_indices)
        
#         if best_feature is None or best_gain <= 0:
#             return {'leaf': True, 'class': np.bincount(y).argmax()}
        
#         # Split the data
#         X_left, y_left = X[best_left_mask], y[best_left_mask]
#         X_right, y_right = X[~best_left_mask], y[~best_left_mask]
        
#         # Recursively build the tree
#         left_subtree = self.fit(X_left, y_left, depth + 1)
#         right_subtree = self.fit(X_right, y_right, depth + 1)
        
#         return {
#             'leaf': False,
#             'feature': best_feature,
#             'threshold': best_threshold,
#             'left': left_subtree,
#             'right': right_subtree
#         }
    
#     def predict_one(self, x, node):
#         if node['leaf']:
#             return node['class']
#         if x[node['feature']] <= node['threshold']:
#             return self.predict_one(x, node['left'])
#         else:
#             return self.predict_one(x, node['right'])
    
#     def predict(self, X, node):
#         X = np.array(X)
#         return np.array([self.predict_one(x, node) for x in X])
    
#     def predict_proba_one(self, x, node):
#         """Predict class probabilities for a single sample (based on leaf distribution)."""
#         if node['leaf']:
#             # Use np.bincount with minlength=self.n_classes to ensure consistent shape
#             class_counts = np.bincount([node['class']], minlength=self.n_classes)
#             return class_counts / np.sum(class_counts)
        
#         if x[node['feature']] <= node['threshold']:
#             return self.predict_proba_one(x, node['left'])
#         else:
#             return self.predict_proba_one(x, node['right'])
    
#     def predict_proba(self, X, node):
#         """Predict class probabilities for all samples."""
#         X = np.array(X)
#         return np.array([self.predict_proba_one(x, node) for x in X])

# class RandomForest:
#     def __init__(self, n_trees=100, max_depth=None, min_samples_split=2, 
#                  max_features='sqrt', random_state=None):
#         self.n_trees = n_trees
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.max_features = max_features
#         self.random_state = random_state
#         self.trees_ = []
#         self.classes_ = None
    
#     def fit(self, X, y):
#         X = np.array(X)
#         y = np.array(y)
        
#         if X.shape[0] != y.shape[0]:
#             raise ValueError(f"X and y must have the same number of samples. "
#                              f"Got {X.shape[0]} for X and {y.shape[0]} for y.")
        
#         n_samples, n_features = X.shape
        
#         if self.random_state is not None:
#             np.random.seed(self.random_state)
        
#         if self.max_features == 'sqrt':
#             self.max_features = int(np.sqrt(n_features))
#         elif self.max_features == 'log2':
#             self.max_features = int(np.log2(n_features))
#         elif isinstance(self.max_features, int):
#             self.max_features = min(self.max_features, n_features)
#         else:
#             self.max_features = n_features
        
#         self.classes_ = np.unique(y)
        
#         self.trees_ = []
#         for _ in range(self.n_trees):
#             indices = np.random.choice(n_samples, n_samples, replace=True)
#             X_boot = X[indices]
#             y_boot = y[indices]
            
#             tree = DecisionTree(
#                 max_depth=self.max_depth,
#                 min_samples_split=self.min_samples_split,
#                 max_features=self.max_features
#             )
#             tree.tree_ = tree.fit(X_boot, y_boot)
#             self.trees_.append(tree)
        
#         return self
    
#     def predict(self, X):
#         X = np.array(X)
#         predictions = np.array([tree.predict(X, tree.tree_) for tree in self.trees_])
#         y_pred = []
#         for i in range(X.shape[0]):
#             votes = predictions[:, i]
#             y_pred.append(np.bincount(votes).argmax())
#         return np.array(y_pred)
    
#     def predict_proba(self, X):
#         X = np.array(X)
#         n_classes = len(self.classes_)
#         proba_sum = np.zeros((X.shape[0], n_classes))
        
#         for tree in self.trees_:
#             proba = tree.predict_proba(X, tree.tree_)
#             # Map probabilities to the correct class indices
#             proba_adjusted = np.zeros((X.shape[0], n_classes))
#             for idx in range(X.shape[0]):
#                 # Since proba[idx] is already aligned with 0 to n_classes-1, adjust for class labels
#                 for c in range(n_classes):
#                     proba_adjusted[idx, c] = proba[idx, c]
#             proba_sum += proba_adjusted
        
#         proba = proba_sum / self.n_trees
#         return proba
    














# class DecisionTree:
#     def __init__(self, max_depth=10, min_samples_split=2):
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.tree = None

#     def entropy(self, y):
#         counter = Counter(y)
#         total = len(y)
#         return -sum((count/total) * log2(count/total) for count in counter.values() if count > 0)

#     def best_split(self, X, y):
#         best_gain = 0
#         best_col = None
#         best_val = None
#         parent_entropy = self.entropy(y)

#         for col in X.columns:
#             values = X[col].unique()
#             for val in values:
#                 left_mask = X[col] <= val
#                 right_mask = ~left_mask

#                 if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
#                     continue

#                 y_left = y[left_mask]
#                 y_right = y[right_mask]
#                 total = len(y)

#                 gain = parent_entropy
#                 gain -= (len(y_left)/total) * self.entropy(y_left)
#                 gain -= (len(y_right)/total) * self.entropy(y_right)

#                 if gain > best_gain:
#                     best_gain = gain
#                     best_col = col
#                     best_val = val

#         return best_col, best_val

#     def build_tree(self, X, y, depth=0):
#         if depth >= self.max_depth or len(y) < self.min_samples_split or len(set(y)) == 1:
#             return Counter(y)

#         col, val = self.best_split(X, y)
#         if col is None:
#             return Counter(y)

#         left_mask = X[col] <= val
#         right_mask = ~left_mask

#         left = self.build_tree(X[left_mask], y[left_mask], depth+1)
#         right = self.build_tree(X[right_mask], y[right_mask], depth+1)

#         return (col, val, left, right)

#     def fit(self, X, y):
#         self.tree = self.build_tree(X, y)

#     def _predict_proba_row(self, row, node):
#         if isinstance(node, Counter):
#             total = sum(node.values())
#             return {cls: count / total for cls, count in node.items()}

#         col, val, left, right = node
#         branch = left if row[col] <= val else right
#         return self._predict_proba_row(row, branch)

#     def predict_proba(self, X):
#         return [self._predict_proba_row(row, self.tree) for _, row in X.iterrows()]

#     def predict(self, X):
#         probs = self.predict_proba(X)
#         return [max(p, key=p.get) for p in probs]

# class RandomForestClassifierCustom:
#     def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, max_features='sqrt'):
#         self.n_estimators = n_estimators
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.max_features = max_features
#         self.trees = []
#         self.features = []

#     def _bootstrap(self, X, y):
#         n_samples = len(X)
#         indices = np.random.choice(n_samples, size=n_samples, replace=True)
#         return X.iloc[indices], y.iloc[indices]

#     def _select_features(self, X):
#         if self.max_features == 'sqrt':
#             n_feats = max(1, int(np.sqrt(X.shape[1])))
#         elif self.max_features == 'log2':
#             n_feats = max(1, int(np.log2(X.shape[1])))
#         elif isinstance(self.max_features, int):
#             n_feats = self.max_features
#         else:
#             n_feats = X.shape[1]
#         return random.sample(list(X.columns), n_feats)

#     def fit(self, X, y):
#         self.trees = []
#         self.features = []

#         for _ in range(self.n_estimators):
#             X_sample, y_sample = self._bootstrap(X, y)
#             feat_subset = self._select_features(X_sample)
#             tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
#             tree.fit(X_sample[feat_subset], y_sample)
#             self.trees.append(tree)
#             self.features.append(feat_subset)

#     def predict_proba(self, X):
#         probs = []

#         for i in range(len(X)):
#             prob_sum = Counter()
#             for tree, feats in zip(self.trees, self.features):
#                 row = X.iloc[[i]][feats]
#                 tree_probs = tree.predict_proba(row)[0]
#                 for cls, p in tree_probs.items():
#                     prob_sum[cls] += p
#             total = sum(prob_sum.values())
#             probs.append({cls: prob_sum[cls]/len(self.trees) for cls in prob_sum})

#         return probs

#     def predict(self, X):
#         probs = self.predict_proba(X)
#         return [max(p, key=p.get) for p in probs]
    












class LogisticRegressionL2_multi:
    """
    Multi-class Logistic Regression with L2 regularization.
    
    Parameters:
    learning_rate (float): Step size for gradient descent (default: 0.01)
    max_iter (int): Maximum number of iterations for gradient descent (default: 1000)
    lambda_reg (float): L2 regularization strength (default: 0.01)
    tol (float): Tolerance for convergence (default: 1e-4)
    class_weights (dict, optional): Weights for each class for cost-sensitive learning (default: None)
    """
    def __init__(self, learning_rate=0.01, max_iter=1000, lambda_reg=0.01, tol=1e-4, class_weights=None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.lambda_reg = lambda_reg
        self.tol = tol
        self.class_weights = class_weights  # For cost-sensitive learning
        self.weights = None
        self.bias = None
        self.classes_ = None
        self.n_classes = None
    
    def softmax(self, z):
        """
        Compute the softmax function for multi-class classification.
        
        Parameters:
        z (np.ndarray): Input of shape (n_samples, n_classes)
        
        Returns:
        np.ndarray: Softmax probabilities of shape (n_samples, n_classes)
        """
        # Subtract the max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def compute_loss(self, X, y, weights, bias):
        """
        Compute the regularized cross-entropy loss for multi-class.
        
        Parameters:
        X (np.ndarray): Data of shape (n_samples, n_features)
        y (np.ndarray): Target values of shape (n_samples,) with class labels
        weights (np.ndarray): Weights of shape (n_features, n_classes)
        bias (np.ndarray): Bias of shape (n_classes,)
        
        Returns:
        float: Loss value
        """
        n_samples = X.shape[0]
        # Compute scores
        z = np.dot(X, weights) + bias
        y_pred = self.softmax(z)
        
        # Avoid log(0) by clipping predictions
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Convert y to one-hot encoding
        y_one_hot = np.zeros((n_samples, self.n_classes))
        for i, label in enumerate(y):
            y_one_hot[i, label - 1] = 1  # Adjust for 0-based indexing
        
        # Cross-entropy loss
        if self.class_weights is None:
            loss = -np.mean(np.sum(y_one_hot * np.log(y_pred), axis=1))
        else:
            # Apply class weights for cost-sensitive learning
            sample_weights = np.array([self.class_weights[label] for label in y])
            loss = -np.mean(sample_weights * np.sum(y_one_hot * np.log(y_pred), axis=1))
        
        # L2 regularization term (excluding bias)
        l2_penalty = (self.lambda_reg / (2 * n_samples)) * np.sum(weights ** 2)
        
        return loss + l2_penalty
    
    def fit(self, X, y):
        """
        Fit the logistic regression model using gradient descent for multi-class.
        
        Parameters:
        X (np.ndarray): Training data of shape (n_samples, n_features)
        y (np.ndarray): Target values of shape (n_samples,) with class labels (e.g., 1, 2, 3)
        """
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        
        # Get unique classes
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        
        # Initialize weights and bias
        self.weights = np.zeros((n_features, self.n_classes))
        self.bias = np.zeros(self.n_classes)
        
        # Gradient descent
        for iteration in range(self.max_iter):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.softmax(z)
            
            # Convert y to one-hot encoding
            y_one_hot = np.zeros((n_samples, self.n_classes))
            for i, label in enumerate(y):
                y_one_hot[i, label - 1] = 1  # Adjust for 0-based indexing
            
            # Compute gradients
            error = y_pred - y_one_hot
            if self.class_weights is not None:
                # Apply class weights
                sample_weights = np.array([self.class_weights[label] for label in y])
                error = error * sample_weights[:, np.newaxis]
            
            grad_weights = (1 / n_samples) * np.dot(X.T, error) + (self.lambda_reg / n_samples) * self.weights
            grad_bias = (1 / n_samples) * np.sum(error, axis=0)
            
            # Update parameters
            weights_new = self.weights - self.learning_rate * grad_weights
            bias_new = self.bias - self.learning_rate * grad_bias
            
            # Check for convergence
            weight_change = np.linalg.norm(weights_new - self.weights)
            if weight_change < self.tol:
                print(f"Converged after {iteration + 1} iterations.")
                break
            
            self.weights = weights_new
            self.bias = bias_new
            
            # Optional: Print loss every 100 iterations
            # if (iteration + 1) % 100 == 0:
            #     loss = self.compute_loss(X, y, self.weights, self.bias)
            #     print(f"Iteration {iteration + 1}, Loss: {loss:.6f}")
        
        return self
    
    def predict_proba(self, X):
        """
        Predict probability estimates for samples.
        
        Parameters:
        X (np.ndarray): Data of shape (n_samples, n_features)
        
        Returns:
        np.ndarray: Predicted probabilities of shape (n_samples, n_classes)
        """
        X = np.array(X)
        z = np.dot(X, self.weights) + self.bias
        return self.softmax(z)
    
    def predict(self, X):
        """
        Predict class labels for samples.
        
        Parameters:
        X (np.ndarray): Data of shape (n_samples, n_features)
        
        Returns:
        np.ndarray: Predicted class labels of shape (n_samples,) with labels (e.g., 1, 2, 3)
        """
        probs = self.predict_proba(X)
        # Return the class with the highest probability, adjusting for 1-based labels
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
        self.classes_ = None  # To store all possible class labels
    
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
        # Store all possible class labels
        self.classes_ = sorted(np.unique(y))  # e.g., [1, 2, 3]
        self.tree = self.build_tree(X, y)
    
    def _predict_proba_row(self, row, node):
        if isinstance(node, Counter):
            total = sum(node.values())
            # Initialize probabilities for all classes
            probs = [0.0] * len(self.classes_)
            for cls, count in node.items():
                # Map class to index (e.g., class 1 -> index 0, class 2 -> index 1, class 3 -> index 2)
                idx = self.classes_.index(cls)
                probs[idx] = count / total if total > 0 else 0.0
            return probs
        
        col, val, left, right = node
        branch = left if row[col] <= val else right
        return self._predict_proba_row(row, branch)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for all samples.
        
        Parameters:
        X (pd.DataFrame): Data to predict on
        
        Returns:
        list: List of lists, where each inner list is [p1, p2, p3] for classes 1, 2, 3
        """
        return [self._predict_proba_row(row, self.tree) for _, row in X.iterrows()]
    
    def predict(self, X):
        probs = self.predict_proba(X)
        # Map the index of the highest probability back to the class label
        return [self.classes_[np.argmax(p)] for p in probs]

class RandomForestClassifierCustom:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.features = []
        self.classes_ = None  # To store all possible class labels
    
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
        # Store all possible class labels
        self.classes_ = sorted(np.unique(y))  # e.g., [1, 2, 3]
        
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
        """
        Predict class probabilities for all samples by averaging across trees.
        
        Parameters:
        X (pd.DataFrame): Data to predict on
        
        Returns:
        list: List of lists, where each inner list is [p1, p2, p3] for classes 1, 2, 3
        """
        probs = []
        
        for i in range(len(X)):
            # Initialize probability sum for all classes
            prob_sum = [0.0] * len(self.classes_)
            for tree, feats in zip(self.trees, self.features):
                row = X.iloc[[i]][feats]
                tree_probs = tree.predict_proba(row)[0]  # Already in [p1, p2, p3] format
                for idx in range(len(self.classes_)):
                    prob_sum[idx] += tree_probs[idx]
            # Average the probabilities
            avg_probs = [p / self.n_estimators for p in prob_sum]
            probs.append(avg_probs)
        
        return probs
    
    def predict(self, X):
        probs = self.predict_proba(X)
        # Map the index of the highest probability back to the class label
        return [self.classes_[np.argmax(p)] for p in probs]