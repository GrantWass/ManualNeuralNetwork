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

def activation_function(x, activation="relu"):
    """
    Apply an activation function to the input.
    
    Args:
        x (numpy.ndarray): Input array.
        activation (str): Type of activation function ("relu", "sigmoid", "tanh", "softmax").
    
    Returns:
        numpy.ndarray: Activated output.
    """
    if activation == "relu":
        return np.maximum(0, x)
    elif activation == "sigmoid":
        return 1 / (1 + np.exp(-x))
    elif activation == "tanh":
        return np.tanh(x)
    elif activation == 'softmax':
        exp_z = np.exp(x - np.max(x))  # To improve numerical stability
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)  # Normalize to sum to 1
    else:
        raise ValueError("Unsupported activation function.")
    
def activation_derivative(A, activation="relu"):
    """
    Compute the derivative of the activation function (used in backpropagation).
    
    Args:
        A (numpy.ndarray): The output of the activation function (A = f(Z)).
        activation (str): The type of activation function (default is 'relu').
    
    Returns:
        numpy.ndarray: The derivative of the activation function.
    """
    if activation == "relu":
        return np.where(A > 0, 1, 0)  # ReLU derivative
    elif activation == "sigmoid":
        return A * (1 - A)  # Sigmoid derivative
    elif activation == "softmax":
        # For softmax, the derivative requires special handling
        # We'll use the Jacobian matrix or simplified versions here
        return A * (1 - A)  # For simplicity, we use this for now
    else:
        raise ValueError(f"Unsupported activation: {activation}")

def loss_function(predictions, targets, type):
    """
    Calculate the loss between predictions and actual targets.
    
    Args:
        predictions (numpy.ndarray): Model predictions .
        targets (numpy.ndarray): Ground truth labels .
        type (string): Type of loss function.
    
    Returns:
        float: Loss value.
    """

    if type == "mse":
        errors = predictions - targets
        squared_errors = errors ** 2
        return np.mean(squared_errors)
    elif type == "cross_entropy":
        # Clip predictions to avoid log(0) (which is undefined)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        # Calculate cross-entropy for each sample
        return -np.sum(targets * np.log(predictions)) / targets.shape[0]
    else:
        raise ValueError("Unsupported loss function type.")


def calculate_accuracy(predictions, targets):
    """
    Calculate the accuracy of predictions compared to targets.
    
    Args:
        predictions (numpy.ndarray): Predicted outputs.
        targets (numpy.ndarray): Actual target outputs.
    
    Returns:
        float: Accuracy value.
    """
    correct = np.sum(predictions == targets)
    total = len(targets)
    return correct / total

