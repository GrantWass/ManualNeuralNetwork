import numpy as np
from utils import activation_function, loss_function, activation_derivative

def compute_gradients(model, input_array, target_array, activations_dict):
    """
    Compute gradients for weights and biases using backpropagation.
    
    Args:
        model (dict): Neural network model containing weights and biases.
        input_array (numpy.ndarray): Input data for the neural network.
        target_array (numpy.ndarray): Ground truth labels.
        activations_dict (dict): Dictionary containing activations and pre-activations.
    
    Returns:
        dict: Gradients for weights and biases.
    """
    gradients = {}
    layer_count = len(model) // 2

    # This will need to change to be more generalized (works for any loss function)

    # Compute the loss gradient for Cross-Entropy Loss
    # Gradient of Cross-Entropy Loss with Softmax: dL/dA = A - Y
    # A is the output (predictions), Y is the target (one-hot encoded)
    loss_gradient = activations_dict[f"A{layer_count}"] - target_array  # Cross-Entropy with Softmax
    
    # Backpropagation (start from the output layer)
    for i in reversed(range(1, layer_count + 1)):  # Iterate over layers in reverse order (backpropagation)
        
        Z = activations_dict[f"Z{i}"]
        A = activations_dict[f"A{i}"]
        W = model[f"W{i}"]
        b = model[f"b{i}"]

        # Compute the gradient of the loss w.r.t. activation (dL/dA)
        if i == layer_count:  # Output layer
            dA = loss_gradient  # Gradient at the output layer
        else:
            # Compute dL/dA for the hidden layers by backpropagating the error
            dA = np.dot(W, dZ) * activation_derivative(A)  # Chain rule (multiply by activation derivative)
        
        # Compute gradients w.r.t. weights and biases
        dW = np.dot(A.T, dA)  # Gradient of the loss w.r.t. weights
        db = np.sum(dA, axis=0, keepdims=True)  # Gradient of the loss w.r.t. biases
        
        # Store the gradients
        gradients[f"W{i}"] = dW
        gradients[f"b{i}"] = db
        
        # Compute dZ (the error for the current layer) for the next layer
        dZ = np.dot(W.T, dA)  # This will be used to propagate the error back to the previous layer
    
    return gradients

def init_model(input_size, hidden_layers, output_size):
    """
    Initialize the neural network model with random weights and biases.
    
    Args:
        input_size (int): Number of input features.
        hidden_layers (list): List of integers representing the number of nodes in each hidden layer.
        output_size (int): Number of output nodes.
    
    Returns:
        dict: Initialized model with weights and biases.
    """
    model = {}
    print("Initializing model...", input_size, hidden_layers, output_size)
    
    layer_sizes = [input_size] + hidden_layers + [output_size]
    
    for i in range(1, len(layer_sizes)):
        # Initialize weights and biases for each layer
        input_size = layer_sizes[i-1]
        output_size = layer_sizes[i]
        weight_matrix = np.random.randn(input_size, output_size)  # Random weights
        bias_vector = np.random.randn(output_size)  # Random biases
        
        # Store weights and biases in the model dictionary
        model[f"W{i}"] = weight_matrix
        model[f"b{i}"] = bias_vector
    
    return model

def forward_propagation(model, input_array, activations=None):
    """
    Perform forward propagation through the model.

    Args:
        model (dict): Neural network model containing weights and biases.
        input_array (numpy.ndarray): Input data (shape: (input_size, batch_size)).
        activations (list): List of activation functions for each layer.

    Returns:
        tuple: Output of the model and a dictionary of intermediate layer activations.
    """
    layer_count = len(model) // 2  

    if activations is None:
        #ReLU for hidden layers, Softmax for output
        activations = ["relu"] * (layer_count - 1) + ["softmax"]

    activations_dict = {}  # Stores activations for each layer
    A = input_array # Input to the first layer of the model (shape: (in_features, sampele_size))
    
    for i in range(1, layer_count + 1):
        W = model[f"W{i}"] # Weight matrix for this layer (shape: (in_features, out_features))
        b = model[f"b{i}"].reshape(-1, 1) # Bias vector for this layer (shape: (out_features, 1))
        activation_type = activations[i - 1]

        # We must transpose W for this calculation to work
        # W transpose is (out_features, in_features) which allows us to matrix multiply W * A
        # Weights * Input + Biases
        Z = np.dot(W.T, A) + b
        A = activation_function(Z, activation=activation_type)  # Apply activation
        
        # Store intermediate values
        activations_dict[f"Z{i}"] = Z
        activations_dict[f"A{i}"] = A

    return A, activations_dict

def back_propagation(model, gradients, learning_rate):
    """
    Perform backpropagation to adjust the model's weights and biases.
    
    Args:
        model (dict): Neural network model containing weights and biases.
        gradients (dict): Gradients for weights and biases computed during training.
        learning_rate (float): Learning rate for updating weights.
    """
    
    layer_count = len(model) // 2

    for i in range(1, layer_count + 1):
        dW = gradients[f"W{i}"]  # Gradient for weights
        db = gradients[f"b{i}"]  # Gradient for biases

        # Apply gradient descent update rule
        model[f"W{i}"] -= learning_rate * dW  # Update weights
        model[f"b{i}"] -= learning_rate * db  # Update biases

def predict(model, input_array):
    """
    Make predictions using the trained model.
    
    Args:
        model (dict): Trained neural network model.
        input_array (numpy.ndarray): Input data for predictions.
    
    Returns:
        numpy.ndarray: Predicted output.
    """
    output, _ = forward_propagation(model, input_array)
    
    return output

def calculate_accuracy(predictions, targets):
    """
    Calculate the accuracy of predictions compared to targets.
    
    Args:
        predictions (numpy.ndarray): Predicted outputs.
        targets (numpy.ndarray): Actual target outputs.
    
    Returns:
        float: Accuracy value.
    """
    
    pass
