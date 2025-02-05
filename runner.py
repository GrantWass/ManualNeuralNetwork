from NeuralNetwork import NeuralNetwork
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from utils import one_hot_encode


iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names).to_numpy()
y_labels = iris.target
y = one_hot_encode(y_labels, num_classes=len(np.unique(y_labels)))
input_size = X.shape[1]
output_size = len(np.unique(y_labels))
hidden_layers = [5, 3]
architecture = [input_size] + hidden_layers + [output_size]


# Initialize model
model = NeuralNetwork(architecture)

# Train model
acc, losses = model.train(X, y, alpha=0.01, epochs=200)
