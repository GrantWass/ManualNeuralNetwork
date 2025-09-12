from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from utils import one_hot_encode
import numpy as np
import pandas as pd

def load_dataset(dataset_name):
    if dataset_name == "iris":
        return load_iris_dataset()
    elif dataset_name == "auto_mpg":
        return load_auto_mpg_dataset()
    else:
        raise ValueError("Unsupported dataset. Choose 'iris' or 'auto_mpg'.")

def load_auto_mpg_dataset():
    # Fetch the Auto MPG dataset
    auto_mpg = fetch_openml('autoMpg', version=1, as_frame=True)
    df = auto_mpg.data
    
    # Select only 4 key features for simplicity: displacement, horsepower, weight, acceleration
    feature_columns = ['displacement', 'horsepower', 'weight', 'acceleration']
    X = df[feature_columns].values
    
    # Target variable is mpg (miles per gallon)
    y = auto_mpg.target.values.reshape(-1, 1)
    
    # Remove any rows with missing values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
    X = X[mask]
    y = y[mask]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    original_train_data = np.hstack((X_train, y_train)).tolist()
    
    # Standardize the input features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train.shape[1], y_train.shape[1], "linear", original_train_data

def load_iris_dataset():
    iris = load_iris()
    X = iris.data
    y = iris.target
    Y = one_hot_encode(y, num_classes=3)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    original_train_data = np.hstack((X_train, Y_train)).tolist()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, Y_train, Y_test, X_train.shape[1], Y_train.shape[1], "softmax", original_train_data