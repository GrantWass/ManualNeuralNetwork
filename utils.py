import pandas as pd
import numpy as np

def normalize_data(data):
    """
    Normalize the input data to a specific range (e.g., 0 to 1).
    
    Args:
        data (pandas.DataFrame): Input data with features as columns.
    
    Returns:
        pandas.DataFrame: Normalized data where the features are scaled to [0, 1].
    """
    return (data - data.min()) / (data.max() - data.min())

def split_data(data, target, train_ratio=0.8):
    """
    Split the data into training and testing sets.
    
    Args:
        data (pandas.DataFrame): Input data (features).
        target (numpy.ndarray or pandas.Series): Target labels.
        train_ratio (float): Proportion of data to use for training.
    
    Returns:
        tuple: 
            - Training data (features) and labels
            - Testing data (features) and labels
    """
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    target = target[data.index]

    # Calculate the split index
    train_size = int(train_ratio * len(data))
    
    # Split the data into training and testing sets
    X_train = data.iloc[:train_size]
    X_test = data.iloc[train_size:]
    
    y_train = target[:train_size]
    y_test = target[train_size:]
    
    return X_train, X_test, y_train, y_test
