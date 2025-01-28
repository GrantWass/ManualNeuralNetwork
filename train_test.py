from model import forward_propagation, back_propagation, compute_gradients
from loss import loss_function, calculate_accuracy
from activation import activation_function
import numpy as np

def train(model, input_array, target_array, epochs, learning_rate):
    # Example of a basic training loop
    for epoch in range(epochs):
        # Perform forward propagation
        output, _ = forward_propagation(model, input_array)
        
        # Compute loss
        loss = loss_function(output, target_array)

        # Backpropagation and weight update
        gradients = compute_gradients(model, input_array, target_array)
        back_propagation(model, gradients, learning_rate)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss}")

    return model

def test(model, input_array, target_array):
    # Perform forward propagation
    predictions, _ = forward_propagation(model, input_array)
    
    # Compute accuracy
    accuracy = calculate_accuracy(predictions, target_array)
    return accuracy
