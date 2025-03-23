import numpy as np
import pandas as pd
from src.models import LinearRegression
from src.metrics import *

def train_val_split(dataframe):

    n_samples = len(dataframe)
    np.random.seed(40)
    indices = np.random.permutation(n_samples)

    split_point = int(n_samples * 0.8)

    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    train_df = dataframe.iloc[train_indices]
    val_df = dataframe.iloc[val_indices]

    return train_df, val_df

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
            X_array = X.values
        # if isinstance(y, pd.Series):
        #     y = y.values

        # Ensure y is 1D
        # y = y.flatten()

        # Get the number of samples
        n_samples = X_array.shape[0]

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
        mae_scores = []
        rmse_scores = []
        models = []

        # Perform k-fold cross-validation
        start_idx = 0
        for fold in range(k):
            # Determine the test indices for this fold
            test_size = fold_sizes[fold]
            test_idx = indices[start_idx:start_idx + test_size]
            train_idx = np.concatenate([indices[:start_idx], indices[start_idx + test_size:]])

            # Split the data into training and test sets for this fold
            X_train, X_val = X_array[train_idx], X_array[test_idx]
            y_train, y_val = y[train_idx], y[test_idx]

            # Convert to pandas DataFrame if X was originally a DataFrame (optional)
            if isinstance(X, pd.DataFrame):
                X_train = pd.DataFrame(X_train, columns=X.columns)
                X_val = pd.DataFrame(X_val, columns=X.columns)

            # Initialize and train the model on the training set
            model = LinearRegression(
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
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val,y_pred)
            rmse = root_mean_squared_error(y_val,y_pred)
            mse_scores.append(mse)
            mae_scores.append(mae)
            rmse_scores.append(rmse)
            models.append(model)

            print(f"Fold {fold + 1}/{k} - Test MSE: {mse:.4f}")

            # Update the start index for the next fold
            start_idx += test_size

        # Compute the average MSE across all folds
        mean_mse = np.mean(mse_scores)
        mean_mae = np.mean(mae_scores)
        mean_rmse = np.mean(rmse_scores)
        print(f"\nAverage Test MSE across {k} folds: {mean_mse:.4f}")

        return {
            'mean_mse': mean_mse,
            'mean_mae': mean_mae,
            'mean_rmse': mean_rmse,
            'mse_scores': mse_scores,
            'models': models
        }


if __name__ == '__main__':
    file_path = 'MachineLearning/tp1/data/processed/cleaned_casas_dev.csv'
    df = pd.read_csv(file_path)

    n_samples = len(df)
    indices = np.random.permutation(n_samples)

    split_point = int(n_samples * 0.8)

    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    train_df = df.iloc[train_indices]
    val_df = df.iloc[val_indices]

    train_df.to_csv('MachineLearning/tp1/data/processed/train_casas_dev.csv', index=False)
    val_df.to_csv('MachineLearning/tp1/data/processed/val_casas_dev.csv', index=False)