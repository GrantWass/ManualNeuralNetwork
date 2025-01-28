import numpy as np

def loss_function(predictions, targets):
    """
    Calculate the loss between predictions and actual targets.
    
    Args:
        predictions (numpy.ndarray): Model predictions.
        targets (numpy.ndarray): Ground truth labels.
    
    Returns:
        float: Loss value.
    """
    return np.mean((predictions - targets) ** 2)

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
