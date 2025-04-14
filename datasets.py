from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.datasets import load_iris
from utils import one_hot_encode
import numpy as np

def load_dataset(dataset_name):
    if dataset_name == "iris":
        return load_iris_dataset()
    elif dataset_name == "mnist":
        return load_mnist_dataset()
    elif dataset_name == "california_housing":
        return load_california_housing_dataset()
    else:
        raise ValueError("Unsupported dataset. Choose 'iris' or 'mnist' or 'california_housing.")

def load_california_housing_dataset():
    # Fetch the dataset
    california = fetch_california_housing()
    X = california.data  # Input features
    y = california.target  # Target variable (median house value)
    
    # Reshape y to be a column vector
    y = y.reshape(-1, 1)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    original_train_data = np.hstack((X_train, y_train)).tolist()
    
    # Standardize the input features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train.shape[1], y_train.shape[1], "linear", original_train_data

def load_mnist_dataset():
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        # Flatten the 28x28 images into 784-dimensional vectors
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        # Normalize pixel values to [0, 1]
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        # One-hot encode the labels
        Y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
        Y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

        original_train_data = np.hstack((X_train, Y_train)).tolist()

        return X_train, X_test, Y_train, Y_test, X_train.shape[1], Y_train.shape[1], "softmax", original_train_data

def load_iris_dataset():
        iris = load_iris()
        X = iris.data
        y = iris.target
        Y = one_hot_encode(y, num_classes=3)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        original_train_data = np.hstack((X_train, Y_train)).tolist()

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, Y_train, Y_test, X_train.shape[1], Y_train.shape[1], "softmax", original_train_data