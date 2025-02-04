import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from utils import normalize_data, one_hot_encode, activation_function, loss_function, generate_wt

iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y_labels = iris.target

X = normalize_data(X).to_numpy()

y = one_hot_encode(y_labels, num_classes=len(np.unique(y_labels)))

# Feedforward function
def f_forward(x, w1, w2):
    z1 = x.dot(w1)  # Hidden layer input
    a1 = activation_function(z1)  # Hidden layer output
    z2 = a1.dot(w2)  # Output layer input
    a2 = activation_function(z2)  # Output layer output
    return a1, a2

# Backpropagation
def back_prop(x, y, w1, w2, alpha):
    a1, a2 = f_forward(x, w1, w2)
    
    # Error in output layer
    d2 = (a2 - y) * (a2 * (1 - a2))  # Derivative of sigmoid
    d1 = (d2.dot(w2.T)) * (a1 * (1 - a1))  # Backprop to hidden layer

    # Gradient updates
    w2_adj = a1.T.dot(d2)
    w1_adj = x.T.dot(d1)

    # Update weights
    w1 -= alpha * w1_adj
    w2 -= alpha * w2_adj

    return w1, w2

# Training function
def train(x, y, w1, w2, alpha=0.01, epochs=100):
    acc, losses = [], []
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(x)):
            a1, a2 = f_forward(x[i], w1, w2)
            total_loss += loss_function(a2, y[i])
            w1, w2 = back_prop(x[i].reshape(1, -1), y[i].reshape(1, -1), w1, w2, alpha)

        avg_loss = total_loss / len(x)
        acc.append(100 - avg_loss * 100)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.5f}, Accuracy={acc[-1]:.2f}%")

    return acc, losses, w1, w2

# Prediction function
def predict(x, w1, w2):
    _, out = f_forward(x, w1, w2)
    return np.argmax(out, axis=1)  # Get the class with highest probability

# **UPDATE THE WEIGHTS** to match the Iris dataset:
w1 = generate_wt(4, 5)  # 4 input features → 5 hidden neurons
w2 = generate_wt(5, 3)  # 5 hidden neurons → 3 output classes

# Train the neural network
acc, losses, w1, w2 = train(X, y, w1, w2, alpha=0.1, epochs=500)

# plotting accuracy
plt.plot(acc)
plt.ylabel('Accuracy')
plt.xlabel("Epochs:")
plt.show()
 
# plotting Loss
plt.plot(losses)
plt.ylabel('Loss')
plt.xlabel("Epochs:")
plt.show()