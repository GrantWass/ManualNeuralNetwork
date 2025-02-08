from utils import calculate_metric
from NeuralNetwork import NeuralNetwork
from datasets import load_dataset

def main():
    # Choose dataset
    dataset = "iris"  # Options: "iris", "mnist", "california"

    # Load dataset
    X_train, X_test, Y_train, Y_test, input_size, output_size = load_dataset(dataset)

    # Define the network architecture
    layer_sizes = [input_size, 4, 3, output_size]
    activations = ["relu", "relu", "softmax"]  # Activations for each layer

    # Create and train the neural network
    nn = NeuralNetwork(layer_sizes, activations)
    # nn.train(X_train, Y_train, epochs=100, learning_rate=0.1)
    epochs = 10
    for epoch in range(epochs):
        result = nn.train_step(X_train, Y_train, learning_rate=0.1)
        print(f"Epoch {epoch}, Loss: {result['loss']}, {list(result.keys())[1]}: {list(result.values())[1]}")

    # Evaluate the network on the test set
    Y_test_hat = nn.forward(X_test)
    test_accuracy = calculate_metric(Y_test_hat, Y_test, activations[-1])
    print(f"Test Accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()