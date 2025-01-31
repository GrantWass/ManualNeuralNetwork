from model import forward_propagation, back_propagation, compute_gradients
from utils import loss_function, calculate_accuracy, activation_function
import numpy as np

def train(model, input_array, target_array, epochs, learning_rate):
    for epoch in range(epochs):
        # Perform forward propagation
        output, activation_dict = forward_propagation(model, input_array)
        
        # Compute loss
        loss = loss_function(output, target_array, "cross_entropy")

        # Backpropagation and weight update
        gradients = compute_gradients(model, input_array, target_array, activation_dict)
        back_propagation(model, gradients, learning_rate)

        if epoch % 1 == 0:
            print(f"Epoch {epoch}: Loss = {loss}")

    return model

def test(model, input_array, target_array):
    # Perform forward propagation
    predictions, _ = forward_propagation(model, input_array)
    
    # Compute accuracy
    accuracy = calculate_accuracy(predictions, target_array)
    return accuracy
