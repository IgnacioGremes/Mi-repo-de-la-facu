import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


#/home/ig/Desktop/prueba/MachineLearning/tp1/data/raw/casas_dev.csv
file_path = 'MachineLearning/tp1/data/processed/cleaned_casas_dev.csv'
df = pd.read_csv(file_path)

plt.figure(figsize=(10, 5))

plt.scatter(df['area'],df['price'])
plt.xlabel('area')
plt.ylabel('price')

plt.show()

plt.figure(figsize=(10, 8))

plt.subplot(2,2,1)
plt.scatter(df['age'],df['price'])
plt.xlabel('age')
plt.ylabel('price')

plt.subplot(2,2,2)
plt.scatter(df['age'],df['area'])
plt.xlabel('age')
plt.ylabel('area')

plt.subplot(2,2,3)
plt.scatter(df['rooms'],df['price'])
plt.xlabel('rooms')
plt.ylabel('price')

plt.subplot(2,2,4)
plt.scatter(df['rooms'],df['area'])
plt.xlabel('rooms')
plt.ylabel('area')
# sb.pairplot(df)
plt.show()

# area vs precio , age vs area y age vs precio (mostrar que se parecen mucho)
# rooms vs area, rooms vs price 

def mean_squared_error(y_true, y_pred):
    """
    Compute the Mean Squared Error (MSE) between true and predicted values.
    
    Parameters:
    y_true (numpy.ndarray): True target values of shape (n_samples,)
    y_pred (numpy.ndarray): Predicted target values of shape (n_samples,)
    
    Returns:
    float: Mean squared error
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    n = len(y_true)
    mse = np.sum((y_true - y_pred) ** 2) / n
    return mse

def mean_value_per_m2(dataframe):
    assert isinstance(dataframe, pd.DataFrame)
    mean_list = []
    for _ , row in dataframe.iterrows():
        mean_list.append(row['price'] / row['area'])
    mean = sum(mean_list) / dataframe.shape[0]
    return mean

file_path = 'MachineLearning/tp1/data/processed/cleaned_m2_casas_dev.csv'
df1 = pd.read_csv(file_path)

print(mean_value_per_m2(df1))