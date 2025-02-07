import numpy as np
import matplotlib.pyplot as plt
from utils import generate_wt, activation_function, loss_function, activation_derivative, calculate_accuracy

class NeuralNetwork:
    """
    A simple feedforward neural network with customizable architecture, activation functions,
    and training using backpropagation with gradient descent.
    """
    def __init__(self, architecture):
        """
        Initializes the neural network with the given architecture.

        Parameters:
        - architecture (list): A list defining the number of neurons in each layer.
                               Example: [4, 5, 3] (4 input neurons → 5 hidden → 3 output)
        """
        self.weights = {}  # Dictionary to store weight matrices
        self.biases = {}   # Dictionary to store bias vectors

        for i in range(len(architecture) - 1):
            self.weights[f"W{i+1}"] = generate_wt(architecture[i], architecture[i+1])  # Random weight initialization
            self.biases[f"b{i+1}"] = np.zeros((1, architecture[i+1]))  # Initialize biases to zero

    def forward_propagation(self, x, activation_types=None):
        """
        Performs forward propagation through the network.
        
        Parameters:
        - x (ndarray): Input data of shape (num_samples, input_dim)
        - activation_types (list, optional): Activation function types for each layer.
                                             Defaults to sigmoid for all layers if None.
        
        Returns:
        - activations (dict): Stores activations and weighted inputs (Z values) for each layer.
        """
        activations = {"A0": x}  # Store activations (input layer is A0)
        if activation_types is None:
            activation_types = ["sigmoid"] * (len(self.weights) - 1) + ["softmax"]

        for i in range(1, len(self.weights) + 1):
            activation_type = activation_types[i - 1]
            z = activations[f"A{i-1}"].dot(self.weights[f"W{i}"]) + self.biases[f"b{i}"]
            activations[f"Z{i}"] = z  # Store raw weighted sum before activation
            activations[f"Z{i}_type"] = activation_type # Store activation type
            activations[f"A{i}"] = activation_function(z, activation=activation_type)  # Apply activation

        return activations

    def back_propagation(self, y, activations):    
        """
        Performs backpropagation to compute gradients of weights and biases.
        
        Parameters:
        - y (ndarray): True labels of shape (num_samples, output_dim)
        - activations (dict): Dictionary containing forward propagation results.
        
        Returns:
        - gradients (dict): Gradients of weights and biases for each layer.
        """
        L = len(self.weights)  # Number of layers
        num_samples = y.shape[0]
        gradients = {}

        # Compute error at the output layer
        if activations[f"Z{L}_type"] == "softmax":
            dA = activations[f"A{L}"] - y
        else:
            dA = (activations[f"A{L}"] - y) * activation_derivative(activations[f"Z{L}"], activation=activations[f"Z{L}_type"])

        # Backpropagate the error
        for i in reversed(range(1, L + 1)):
            dW = (1 / num_samples) * activations[f"A{i-1}"].T.dot(dA)  # Compute weight gradient
            db = (1 / num_samples) * np.mean(dA, axis=0, keepdims=True)  # Compute bias gradient
            gradients[f"dW{i}"] = dW
            gradients[f"db{i}"] = db

            # Propagate the error backward to the previous layer
            if i > 1:
                dA = dA.dot(self.weights[f"W{i}"].T) * activation_derivative(activations[f"Z{i-1}"], activation=activations[f"Z{i-1}_type"])

        return gradients

    def update_parameters(self, gradients, alpha):
        """
        Updates the weights and biases using the computed gradients.
        
        Parameters:
        - gradients (dict): Dictionary containing gradients of weights and biases.
        - alpha (float): Learning rate.
        """

        for i in range(1, len(self.weights) + 1):
            self.weights[f"W{i}"] -= alpha * gradients[f"dW{i}"]  # Update weights
            self.biases[f"b{i}"] -= alpha * gradients[f"db{i}"]  # Update biases

    def train(self, x, y, alpha=0.01, epochs=100, type="sgd", batch_size=5):
        """
        Trains the neural network using gradient descent.
        
        Parameters:
        - x (ndarray): Training data of shape (num_samples, input_dim)
        - y (ndarray): One-hot encoded labels of shape (num_samples, output_dim)
        - alpha (float): Learning rate.
        - epochs (int): Number of training epochs.
        
        Returns:
        - acc (list): List of accuracy values over epochs.
        - losses (list): List of loss values over epochs.
        """
        acc = []
        losses = []
        for epoch in range(epochs):
            avg_loss = 0

            if type == "sgd":
                indices = np.arange(len(x))
                np.random.shuffle(indices)
                for i in range(0, len(x), batch_size):
                    batch_indices = indices[i:i + batch_size]
                    x_batch = x[batch_indices]
                    y_batch = y[batch_indices]
                    loss = self.training_cycle(x_batch, y_batch, alpha)
                    avg_loss += loss
                avg_loss /= (len(x) // batch_size)
            else:
                avg_loss = self.training_cycle(x, y, alpha)  
            
            predictions = self.forward_propagation(x)[f"A{len(self.weights)}"]
            accuracy = calculate_accuracy(predictions, y)
            acc.append(accuracy)
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}: Loss={avg_loss:.5f}, Accuracy={acc[-1]:.2f}%")

        self.plot_metrics(acc, losses)
        return acc, losses
    

    def training_cycle(self, x, y, alpha):
        """
        Trains the neural network using gradient descent.
        
        Parameters:
        - x (ndarray): Training data of shape
        - y (ndarray): One-hot encoded labels of shape
        - alpha (float): Learning rate.

        """
        activations = self.forward_propagation(x)  
        loss = loss_function(activations[f"A{len(self.weights)}"], y)  
        gradients = self.back_propagation(y, activations)  
        self.update_parameters(gradients, alpha)  
        return loss  

    def plot_metrics(self, acc, losses):

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Accuracy')
        plt.ylabel('Accuracy (%)')
        plt.xlabel("Epochs")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(losses, label='Loss', color='red')
        plt.ylabel('Loss')
        plt.xlabel("Epochs")
        plt.legend()

        plt.show()

# class NeuronLayer:
#     def __init__(self, input_size, output_size):
#         self.weights = generate_wt(input_size, output_size)
#         self.biases = np.zeros((1, output_size))

# class NeuralNetwork:
#     def __init__(self, architecture):
#         self.layers = []
#         for i in range(len(architecture) - 1):
#             self.layers.append(NeuronLayer(architecture[i], architecture[i+1]))
    