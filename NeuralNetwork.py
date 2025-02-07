from utils import loss_function, calculate_metric, plot_metrics
from NeuronLayer import NeuronLayer

class NeuralNetwork:
    def __init__(self, layer_sizes, activations):
        # Initialize the neural network with a list of layer sizes and activation functions
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            # Create each layer with the appropriate input and output sizes
            self.layers.append(NeuronLayer(layer_sizes[i], layer_sizes[i+1], activations[i]))

    def forward(self, X):
        A = X  # Start with the input data
        for layer in self.layers:
            A = layer.forward(A)  # Pass the output of one layer as input to the next
        return A  # Return the final output of the network

    def backward(self, X, Y, Y_hat):
        # Perform the backward pass through all layers
        m = X.shape[0]  # Number of samples in the batch
        dA = Y_hat - Y  # Derivative of cross-entropy loss w.r.t. softmax output
        
        # Iterate through layers in reverse order
        for i in reversed(range(len(self.layers))):
            prev_A = X if i == 0 else self.layers[i-1].A  # Input to the current layer
            # Compute gradients for the current layer
            # Includes Y for softmax layer
            dA = self.layers[i].backward(dA, prev_A, Y if i == len(self.layers)-1 else None)

    def update_parameters(self, learning_rate):
        # Update the weights and biases of each layer using gradient descent
        for layer in self.layers:
            layer.weights -= learning_rate * layer.dW  # Update weights
            layer.biases -= learning_rate * layer.db  # Update biases

    def train(self, X, Y, epochs, learning_rate, type=None):
        if type is None:
            type = "mse" if self.layers[-1].activation == "linear" else "cross-entropy"

        losses = [] 
        accuracies = [] 
        for epoch in range(epochs):
            Y_hat = self.forward(X)  # Forward pass
            loss = loss_function(Y_hat, Y, type)  # Compute loss
            self.backward(X, Y, Y_hat)  # Backward pass
            self.update_parameters(learning_rate)  # Update parameters

            losses.append(loss) 
            if epoch % 2 == 0:
                metric_value = calculate_metric(Y_hat, Y, self.layers[-1].activation)
                if self.layers[-1].activation == "softmax":
                    accuracies.append(metric_value)
                    print(f"Epoch {epoch}, Loss: {loss}, Accuracy: {metric_value}%")
                else:  # Regression case
                    accuracies.append(metric_value)
                    print(f"Epoch {epoch}, Loss: {loss}, MAE: {metric_value}")
        
        plot_metrics(losses, accuracies, epochs)

    def train_step(self, X, Y, learning_rate, loss_type=None):
        # Determine loss type based on the output activation
        if loss_type is None:
            loss_type = "mse" if self.layers[-1].activation == "linear" else "cross-entropy"

        Y_hat = self.forward(X)
        loss = loss_function(Y_hat, Y, loss_type)
        self.backward(X, Y, Y_hat)
        self.update_parameters(learning_rate)

        metric_value = calculate_metric(Y_hat, Y, self.layers[-1].activation)
        metric_name = "Accuracy" if self.layers[-1].activation == "softmax" else "MAE"

        layer_details = []
        for layer in self.layers:
            layer_details.append({
                "weights": layer.weights,
                "biases": layer.biases,
                "Z": layer.Z,
                "A": layer.A,
                "dW": layer.dW,
                "db": layer.db,
                "dZ": layer.dZ,
                "activation": layer.activation
            })

        return {
            "loss": loss,
            metric_name.lower(): metric_value,
            "layers": layer_details  
        }

