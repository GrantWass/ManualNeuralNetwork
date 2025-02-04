import numpy as np
from utils import activation_function, loss_function, activation_derivative, generate_wt

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

    # Compute the gradient of the loss w.r.t. output (assumes softmax activation with cross-entropy loss)
    loss_gradient = activations_dict[f"A{layer_count}"] - target_array  

    dA = loss_gradient  # Start with output layer error

    # Backpropagation loop
    for i in reversed(range(1, layer_count + 1)):
        Z = activations_dict[f"Z{i}"]
        A_prev = activations_dict[f"A{i-1}"] if i > 1 else input_array  # Input is A0
        W = model[f"W{i}"]

        # Compute dZ using activation derivative
        dZ = dA * activation_derivative(Z)

        # Compute gradients for weights and biases
        dW = np.dot(A_prev, dZ.T) / input_array.shape[1]  # Normalize by batch size
        db = np.mean(dZ, axis=1, keepdims=True)

        # Store gradients
        gradients[f"W{i}"] = dW
        gradients[f"b{i}"] = db

        # Propagate error backward to the previous layer
        dA = np.dot(W, dZ)

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
        weight_matrix = generate_wt(input_size, output_size)  # Random weights
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
        activations = ["sigmoid"] * (layer_count)

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
    predicted_classes = np.argmax(output, axis=0)
    
    return predicted_classes

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
        db = gradients[f"b{i}"].reshape(-1,)  # Gradient for biases

        # Apply gradient descent update rule
        model[f"W{i}"] -= learning_rate * dW  # Update weights
        model[f"b{i}"] -= learning_rate * db  # Update biases
    return 
