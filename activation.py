import numpy as np

def activation_function(x, activation="relu"):
    """
    Apply an activation function to the input.
    
    Args:
        x (numpy.ndarray): Input array.
        activation (str): Type of activation function ("relu", "sigmoid", "tanh").
    
    Returns:
        numpy.ndarray: Activated output.
    """
    if activation == "relu":
        return np.maximum(0, x)
    elif activation == "sigmoid":
        return 1 / (1 + np.exp(-x))
    elif activation == "tanh":
        return np.tanh(x)
    else:
        raise ValueError("Unsupported activation function.")
