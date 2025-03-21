import numpy as np
import pandas as pd
from preprocessing import normalize
from metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

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





# El que funciona



# class LinearRegression:
#     def __init__(self, method='gradient_descent', learning_rate=0.01, epochs=1000):
#         self.method = method
#         self.learning_rate = learning_rate
#         self.epochs = epochs
#         self.theta = None
#         self.feature_names = None
    
#     def fit(self, X, y, feature_names=None):
#         # Convert X to NumPy array if it's a pandas DataFrame
#         if hasattr(X, 'values'):
#             X = X.values

#         # Ensure X_new is 2D (even if it's a single feature)
#         if X.ndim == 1:
#             X = X.reshape(-1, 1)

#         # Convert y to NumPy array if it's a pandas Series
#         if hasattr(y, 'values'):
#             y = y.values

#         X = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
#         self.feature_names = ['bias'] + (feature_names if feature_names else [f'feature_{i}' for i in range(X.shape[1] - 1)])
        
#         if self.method == 'gradient_descent':
#             self._fit_gradient_descent(X, y)
#         elif self.method == 'pseudo_inverse':
#             self._fit_pseudo_inverse(X, y)
#         else:
#             raise ValueError("Invalid method. Choose 'gradient_descent' or 'pseudo_inverse'")
    
#     def _fit_gradient_descent(self, X, y):
#         m, n = X.shape
#         self.theta = np.zeros(n)
#         for _ in range(self.epochs):
#             gradients = (2/m) * X.T @ (X @ self.theta - y)
#             self.theta -= self.learning_rate * gradients
    
#     def _fit_pseudo_inverse(self, X, y):
#         self.theta = np.linalg.pinv(X) @ y  # Compute the normal equation
    
#     def predict(self, X):
#         # Convert X_new to NumPy array if it's a pandas DataFrame
#         if hasattr(X, 'values'):
#             X = X.values
        
#         # Ensure X_new is 2D (even if it's a single feature)
#         if X.ndim == 1:
#             X = X.reshape(-1, 1)

#         X = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
#         return X @ self.theta
    
#     def print_coefficients(self):
#         if self.theta is None or self.feature_names is None:
#             print("Model has not been trained yet.")
#         else:
#             for name, coef in zip(self.feature_names, self.theta):
#                 print(f"{name}: {coef}")








# class LinearRegressionn:
#     def __init__(self, method='gradient_descent', learning_rate=0.01, epochs=1000):
#         self.method = method
#         self.learning_rate = learning_rate
#         self.epochs = epochs
#         self.theta = None
#         self.feature_names = None
    
#     def fit(self, X, y):
#         # Determine feature names based on the type of X
#         if isinstance(X, pd.DataFrame):
#             # Extract column names from DataFrame
#             feature_names = list(X.columns)
#         else:
#             # Convert X to NumPy array if it's not already
#             if hasattr(X, 'values'):
#                 X = X.values
#             # Ensure X is 2D (even if it's a single feature)
#             if X.ndim == 1:
#                 X = X.reshape(-1, 1)
#             # Use default feature names if X is not a DataFrame
#             feature_names = [f'feature_{i}' for i in range(X.shape[1])]

#         # Convert y to NumPy array if it's a pandas Series
#         if hasattr(y, 'values'):
#             y = y.values

#         # Convert X to NumPy array for computation
#         X = X.values if isinstance(X, pd.DataFrame) else X

#         # Add bias term (column of ones)
#         X = np.c_[np.ones((X.shape[0], 1)), X]

#         # Set feature names: 'bias' for the intercept, followed by the feature names
#         self.feature_names = ['bias'] + feature_names
        
#         if self.method == 'gradient_descent':
#             self._fit_gradient_descent(X, y)
#         elif self.method == 'pseudo_inverse':
#             self._fit_pseudo_inverse(X, y)
#         else:
#             raise ValueError("Invalid method. Choose 'gradient_descent' or 'pseudo_inverse'")
    
#     def _fit_gradient_descent(self, X, y):
#         m, n = X.shape
#         self.theta = np.zeros(n)
#         for _ in range(self.epochs):
#             gradients = (2/m) * X.T @ (X @ self.theta - y)
#             self.theta -= self.learning_rate * gradients
    
#     def _fit_pseudo_inverse(self, X, y):
#         self.theta = np.linalg.pinv(X) @ y  # Compute the normal equation
    
#     def predict(self, X):
#         # Convert X to NumPy array if it's a pandas DataFrame
#         if hasattr(X, 'values'):
#             X = X.values
        
#         # Ensure X is 2D (even if it's a single feature)
#         if X.ndim == 1:
#             X = X.reshape(-1, 1)

#         # Add bias term
#         X = np.c_[np.ones((X.shape[0], 1)), X]
#         return X @ self.theta
    
#     def print_coefficients(self):
#         if self.theta is None or self.feature_names is None:
#             print("Model has not been trained yet.")
#         else:
#             print("Model Coefficients:")
#             for name, coef in zip(self.feature_names, self.theta):
#                 print(f"{name}: {coef:.4f}")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegressionn:
    def __init__(self, method='gradient_descent', learning_rate=0.01, epochs=1000, L1=0.0, L2=0.0):
        """
        Initialize the linear regression model with L1 and L2 regularization.
        
        Parameters:
        X (numpy.ndarray or pandas.DataFrame): Feature matrix of shape (n_samples, n_features)
        y (numpy.ndarray or pandas.Series): Target vector of shape (n_samples,)
        method (str): Training method ('gradient_descent' or 'pseudo_inverse')
        learning_rate (float): Learning rate for gradient descent
        epochs (int): Number of iterations for gradient descent
        L1 (float): L1 regularization coefficient (default 0.0)
        L2 (float): L2 regularization coefficient (default 0.0)
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
            if hasattr(X, 'values'):
                X = X.values
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

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
        self.coef = np.zeros(n)  # Initialize coefficients to zeros
        for _ in range(self.epochs):
            # Compute gradients for the MSE term
            gradients = (2/m) * X.T @ (X @ self.coef - y)
            
            # Add L1 regularization term: L1 * sign(w)
            if self.L1 > 0:
                reg_grad_l1 = self.L1 * np.sign(self.coef)
                reg_grad_l1[0] = 0  # Do not regularize the bias term
                gradients += reg_grad_l1
            
            # Add L2 regularization term: 2 * L2 * w
            if self.L2 > 0:
                reg_grad_l2 = 2 * self.L2 * self.coef
                reg_grad_l2[0] = 0  # Do not regularize the bias term
                gradients += reg_grad_l2
            
            # Update coefficients
            self.coef -= self.learning_rate * gradients
    
    def _fit_pseudo_inverse(self, X, y):
        if self.L1 > 0:
            raise ValueError("Pseudo-inverse method does not support L1 regularization. Use gradient descent.")
        if self.L2 > 0:
            # Analytical solution for L2 regularization (Ridge regression)
            n_features = X.shape[1]
            I = np.eye(n_features)
            I[0, 0] = 0  # Do not regularize the bias term
            self.coef = np.linalg.inv(X.T @ X + self.L2 * I) @ X.T @ y
        else:
            # Standard pseudo-inverse (no regularization)
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
    
    def mean_squared_error(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        n = len(y_true)
        mse = np.mean((y_true - y_pred) ** 2)
        return mse




    
# hacer los validation y medir el mean squared error

# Read train file

file_path = 'MachineLearning/tp1/data/processed/train_cleaned_casas_dev.csv'
df_train = pd.read_csv(file_path)
X_train = df_train.drop(columns=['price'])
Y_train = df_train['price']

# Read validation file

file_path = 'MachineLearning/tp1/data/processed/val_cleaned_casas_dev.csv'
df_val = pd.read_csv(file_path)
X_val = df_val.drop(columns=['price'])
Y_val = df_val['price']

# Read amanda file

file_path = 'MachineLearning/tp1/data/processed/cleaned_Amanda.csv'
df_amanda = pd.read_csv(file_path)

# Regresion with only area

regresion = LinearRegressionn(method='pseudo_inverse')
regresion.fit(X_train['area'],Y_train)
print('')

regresion.print_coefficients()

pred_val = regresion.predict(X_val['area'])
print(f'mean squared error: {mean_squared_error(Y_val, pred_val)}')
print(Y_val)
print('----------------------------------------------')
print(pred_val)

pred_amanda = regresion.predict(df_amanda['area'])
print(pred_amanda)


# print('Resultados de sklearn')
# reg = LinearRegression().fit(X_train['area'].to_frame(),Y_train)
# print(reg.score(X_val['area'].to_frame(),Y_val))

# print(reg.predict(df_amanda['area'].to_frame()))





# Regresion

regresion = LinearRegressionn(method='pseudo_inverse')
regresion.fit(X_train,Y_train)
print('')

regresion.print_coefficients()

pred_val = regresion.predict(X_val)
print(f'mean squared error: {mean_squared_error(Y_val, pred_val)}')
print(Y_val)
print('----------------------------------------------')
print(pred_val)

pred_amanda = regresion.predict(df_amanda)
print(pred_amanda)









# Read own featured train file

file_path = 'MachineLearning/tp1/data/processed/own_featured_train_casas_dev.csv'
df = pd.read_csv(file_path)
X_train = df.drop(columns=['price'])
Y_train = df['price']

# Read own featured validation file

file_path = 'MachineLearning/tp1/data/processed/own_featured_val_casas_dev.csv'
df_val = pd.read_csv(file_path)
X_val = df_val.drop(columns=['price'])
Y_val = df_val['price']

# Read own featured amanda file

file_path = 'MachineLearning/tp1/data/processed/own_featured_Amanda.csv'
df_amanda = pd.read_csv(file_path)


# Regresion with own features

regresion = LinearRegressionn(method='pseudo_inverse')
regresion.fit(X_train,Y_train)

regresion.print_coefficients()

pred_val = regresion.predict(X_val)
print(f'mean squared error: {mean_squared_error(Y_val, pred_val)}')
print(Y_val)
print('----------------------------------------------')
print(pred_val)

pred_amanda = regresion.predict(df_amanda)
print(pred_amanda)



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




# Regresion with L2

# Read own featured train file

file_path = 'MachineLearning/tp1/data/processed/own_featured_train_casas_dev.csv'
df = pd.read_csv(file_path)
X_train = df.drop(columns=['price'])
Y_train = df['price']

# Read own featured validation file

file_path = 'MachineLearning/tp1/data/processed/own_featured_val_casas_dev.csv'
df_val = pd.read_csv(file_path)
X_val = df_val.drop(columns=['price'])
Y_val = df_val['price']

# Read own featured amanda file

file_path = 'MachineLearning/tp1/data/processed/own_featured_Amanda.csv'
df_amanda = pd.read_csv(file_path)

# Range of L2 values to test
l2_values = np.logspace(-4, 2, 50)  # From 10^-4 to 10^2
# l2_values = [-3,-2,-1,1,2,3,4,5,6,7,8,9,10]
# Store coefficients for each L2 value
coefficients = []

# Train the model for each L2 value (L1 = 0)
for l2 in l2_values:
    model = LinearRegressionn(method='pseudo_inverse', learning_rate=0.01, epochs=1000, L1=0.0, L2=l2)
    model.fit(X_train,Y_train)
    coefficients.append(model.coef)

# Convert coefficients to a NumPy array for plotting
coefficients = np.array(coefficients)

# Plot the coefficients vs. L2
plt.figure(figsize=(10, 6))
for i, feature_name in enumerate(model.feature_names):
    plt.plot(l2_values, coefficients[:, i], label=feature_name)

plt.xscale('log')
plt.xlabel('L2 Regularization Coefficient (L2)')
plt.ylabel('Coefficient Value (w*)')
plt.title('Optimal Weights vs. L2 Regularization Coefficient')
plt.legend()
plt.grid(True)
plt.show()

# Elegir lambda fundamentado
print('')
print('Elegir lambda')
model = LinearRegressionn(method='pseudo_inverse', learning_rate=0.01, epochs=1000, L1=0.0, L2=6)
model.fit(X_train,Y_train)

pred_val = model.predict(X_val)
print(f'mean squared error: {mean_squared_error(Y_val, pred_val)}')

model.print_coefficients()
print(Y_val)
print('----------------------------------------------')
print(pred_val)

# Barrido de lambdas para mejor MSE

# Range of L2 values to test
l2_values = np.logspace(-4, 2, 50)  # From 10^-4 to 10^2

# Store coefficients for each L2 value
ecms = []

# Train the model for each L2 value (L1 = 0)
for l2 in l2_values:
    model = LinearRegressionn(method='pseudo_inverse', learning_rate=0.01, epochs=1000, L1=0.0, L2=l2)
    model.fit(X_train,Y_train)
    pred_val = model.predict(X_val)
    ecms.append(mean_squared_error(Y_val, pred_val))


# Plot the coefficients vs. L2
plt.figure(figsize=(10, 6))
# for i, feature_name in enumerate(model.feature_names):
plt.plot(l2_values, ecms)

# plt.xscale('log')
plt.xlabel('L2 Regularization Coefficient (L2)')
plt.ylabel('Error Cuadratico Medio (ECM)')
# plt.title('Optimal Weights vs. L2 Regularization Coefficient')
plt.legend()
plt.grid(True)
plt.show()



# Regresion with L1

file_path = 'MachineLearning/tp1/data/processed/normalized_own_featured_train_casas_dev.csv'
df = pd.read_csv(file_path)
X_train = df.drop(columns=['price'])
Y_train = df['price']

# Read own featured validation file

file_path = 'MachineLearning/tp1/data/processed/normalized_own_featured_val_casas_dev.csv'
df_val = pd.read_csv(file_path)
X_val = df_val.drop(columns=['price'])
Y_val = df_val['price']

# Range of L1 values to test
l1_values = np.logspace(-4, 2, 50)  # From 10^-4 to 10^2

# Store coefficients for each L1 value
ecms = []

# Train the model for each L1 value (L1 = 0)
for l1 in l1_values:
    model = LinearRegressionn(method='gradient_descent', learning_rate=0.01, epochs=1000, L1=l1, L2=0.0)
    model.fit(X_train,Y_train)
    ecms.append(model.coef)

# Convert coefficients to a NumPy array for plotting
ecms = np.array(ecms)

# Plot the coefficients vs. L1
plt.figure(figsize=(10, 6))
for i, feature_name in enumerate(model.feature_names):
    plt.plot(l1_values, ecms[:, i], label=feature_name)

plt.xscale('log')
plt.xlabel('L1 Regularization Coefficient (L1)')
plt.ylabel('Coefficient Value (w*)')
plt.title('Optimal Weights vs. L1 Regularization Coefficient')
plt.legend()
plt.grid(True)
plt.show()



























def cross_validate_linear_regression(X, y, k=5, method='gradient_descent', learning_rate=0.01, epochs=1000, L1=0.0, L2=0.0):
    """
    Perform k-fold cross-validation on a dataset using the LinearRegressionn class, without sklearn.
    
    Parameters:
    X (numpy.ndarray or pandas.DataFrame): Feature matrix of shape (n_samples, n_features)
    y (numpy.ndarray or pandas.Series): Target vector of shape (n_samples,)
    k (int): Number of folds for cross-validation (default 5)
    method (str): Training method for LinearRegressionn ('gradient_descent' or 'pseudo_inverse')
    learning_rate (float): Learning rate for gradient descent
    epochs (int): Number of iterations for gradient descent
    L1 (float): L1 regularization coefficient
    L2 (float): L2 regularization coefficient
    
    Returns:
    dict: A dictionary containing:
        - 'mean_mse': Average MSE across all folds
        - 'mse_scores': List of MSE scores for each fold
        - 'models': List of trained LinearRegressionn models for each fold
    """
    # Convert X and y to NumPy arrays if they are pandas objects
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    # Ensure y is 1D
    # y = y.flatten()

    # Get the number of samples
    n_samples = X.shape[0]

    # Shuffle the indices
    indices = np.arange(n_samples)
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(indices)

    # Calculate the size of each fold
    fold_size = n_samples // k
    fold_sizes = [fold_size] * k
    remaining = n_samples % k
    for i in range(remaining):
        fold_sizes[i] += 1  # Distribute the remaining samples

    # Initialize lists to store results
    mse_scores = []
    models = []

    # Perform k-fold cross-validation
    start_idx = 0
    for fold in range(k):
        # Determine the test indices for this fold
        test_size = fold_sizes[fold]
        test_idx = indices[start_idx:start_idx + test_size]
        train_idx = np.concatenate([indices[:start_idx], indices[start_idx + test_size:]])

        # Split the data into training and test sets for this fold
        X_train, X_val = X[train_idx], X[test_idx]
        y_train, y_val = y[train_idx], y[test_idx]

        # Convert to pandas DataFrame if X was originally a DataFrame (optional)
        if isinstance(X, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=X.columns)
            X_val = pd.DataFrame(X_val, columns=X.columns)

        # Initialize and train the model on the training set
        model = LinearRegressionn(
            method=method,
            learning_rate=learning_rate,
            epochs=epochs,
            L1=L1,
            L2=L2
        )

        model.fit(X_train,y_train)
        # Make predictions on the test set
        y_pred = model.predict(X_val)

        # Compute MSE for this fold
        mse = model.mean_squared_error(y_val, y_pred)
        mse_scores.append(mse)
        models.append(model)

        print(f"Fold {fold + 1}/{k} - Test MSE: {mse:.4f}")

        # Update the start index for the next fold
        start_idx += test_size

    # Compute the average MSE across all folds
    mean_mse = np.mean(mse_scores)
    print(f"\nAverage Test MSE across {k} folds: {mean_mse:.4f}")

    return {
        'mean_mse': mean_mse,
        'mse_scores': mse_scores,
        'models': models
    }


file_path = 'MachineLearning/tp1/data/processed/cleaned_casas_dev.csv'
df = pd.read_csv(file_path)
X = df.drop(columns=['price'])
X = X.drop(columns=['area_units'])
Y = df['price']


l2_values = np.logspace(-4, 2, 50)  # From 10^-4 to 10^2

ecms = []

for l2 in l2_values:
    cv_results = cross_validate_linear_regression(
        X, Y,
        k=5,
        method='pseudo_inverse',
        learning_rate=0.01,
        epochs=1000,
        L1=0.0,
        L2=l2
    )
    ecms.append(cv_results['mean_mse'])

# Plot the coefficients vs. L2
plt.figure(figsize=(10, 6))
# for i, feature_name in enumerate(model.feature_names):
plt.plot(l2_values, ecms)

# plt.xscale('log')
plt.xlabel('L2 Regularization Coefficient (L2)')
plt.ylabel('Error Cuadratico Medio (ECM)')
# plt.title('Optimal Weights vs. L2 Regularization Coefficient')
# plt.legend()
plt.grid(True)
plt.show()