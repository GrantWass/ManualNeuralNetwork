import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.datasets import mnist

# Save Iris dataset
def save_iris_csv():
    iris = load_iris()
    df_iris = pd.DataFrame(data=np.c_[iris.data, iris.target], columns=iris.feature_names + ["label"])
    df_iris.to_csv("iris.csv", index=False)

# Save MNIST dataset
def save_mnist_csv():
    # Load MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Flatten images (28x28 → 784 pixels)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Create DataFrames
    df_train = pd.DataFrame(X_train_flat)
    df_train.insert(0, "label", y_train)  # Insert labels as first column

    df_test = pd.DataFrame(X_test_flat)
    df_test.insert(0, "label", y_test)

    # Combine train & test sets
    df_mnist = pd.concat([df_train, df_test], ignore_index=True)

    # Save to CSV
    df_mnist.to_csv("mnist.csv", index=False)

# Run both functions
save_iris_csv()
save_mnist_csv()

print("All datasets saved!")
