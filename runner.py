from NeuralNetwork import NeuralNetwork
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from utils import one_hot_encode
from tensorflow.keras.datasets import mnist

# iris = load_iris()
# X_train = pd.DataFrame(data=iris.data, columns=iris.feature_names).to_numpy()
# y_labels = iris.target
# y_train = one_hot_encode(y_labels, num_classes=len(np.unique(y_labels)))
# output_size = len(np.unique(y_labels))
# hidden_layers = [4, 3]


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0  # Flatten 28x28 → 784
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
output_size = 10  # Digits 0-9
y_train = one_hot_encode(y_train, num_classes=10)
y_test = one_hot_encode(y_test, num_classes=10)
hidden_layers = [128, 64]

input_size = X_train.shape[1]
architecture = [input_size] + hidden_layers + [output_size]

model = {}

# Initialize model
model = NeuralNetwork(architecture)

# Train model
acc, losses = model.train(X_train, y_train, alpha=0.01, epochs=10)

# test_acc, _ = model.train(X_test, y_test, alpha=0.01, epochs=1)
# print(f"Final Test Accuracy: {test_acc[-1]:.2f}%")
