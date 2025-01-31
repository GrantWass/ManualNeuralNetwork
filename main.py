from model import init_model, predict
from train_test import train, test
from utils import normalize_data, split_data
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

def main():
    # Load the Iris dataset
    iris = load_iris()

    # Set up Iris
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    target_array = iris.target

    # Normalize the data
    data = normalize_data(iris_df)

    # Split the data
    train_data, test_data, train_labels, test_labels = split_data(data, target_array, train_ratio=0.8)

    # Initialize the model
    input_size = train_data.shape[1]
    hidden_layers = [4, 3] # Example
    output_size = len(np.unique(target_array))
    model = init_model(input_size, hidden_layers, output_size)

    # Train the model
    epochs = 30
    learning_rate = 0.01
    trained_model = train(model, train_data.T, train_labels, epochs, learning_rate)

    # Test the model
    accuracy = test(trained_model, test_data, test_labels)
    print(f"Test Accuracy: {accuracy:.2f}")

    # Make predictions
    predictions = predict(trained_model, test_data)
    print(f"Predictions: {predictions[:5]}")
    print(f"Actual Labels: {test_labels[:5]}")

if __name__ == "__main__":  
    main()
