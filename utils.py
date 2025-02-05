import pandas as pd
import numpy as np

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

def activation_function(x, activation="sigmoid"):
    """
    Apply an activation function to the input.
    
    Args:
        x (numpy.ndarray): Input array.
        activation (str): Type of activation function ("relu", "sigmoid", "tanh", "softmax").
    
    Returns:
        numpy.ndarray: Activated output.
    """
    if activation == "sigmoid":
        return 1 / (1 + np.exp(-x))
    elif activation == "relu":
        return np.maximum(0, x)
    elif activation == "tanh":
        return np.tanh(x)
    elif activation == "softmax":
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Numerical stability
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

def activation_derivative(Z, activation, Y=None):
    """
    Compute the derivative of the activation function.
    
    Args:
        Z (numpy.ndarray): Input before activation.
        activation (str): Type of activation function.
    
    Returns:
        numpy.ndarray: Derivative of the activation function.
    """
    if activation == "sigmoid":
        sigmoid = 1 / (1 + np.exp(-Z))
        return sigmoid * (1 - sigmoid)
    elif activation == "relu":
        return np.where(Z > 0, 1, 0)
    elif activation == "tanh":
        return 1 - np.tanh(Z) ** 2
    elif activation == "softmax":
        softmax = activation_function(Z, activation="softmax")
        if Y is not None:
            return softmax - Y  # Correct gradient for backpropagation
        raise ValueError("Softmax derivative requires Y (one-hot labels).")
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

def loss_function(predictions, targets, type= "mse"):
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

def normalize_data(data):
    return (data - data.min()) / (data.max() - data.min())

def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

def generate_wt(input, ouput):
    return np.random.randn(input, ouput)
