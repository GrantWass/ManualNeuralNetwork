from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from utils import one_hot_encode
from NeuralNetwork import NeuralNetwork

app = FastAPI()

# Load dataset (Iris for example)
iris = load_iris()
X_train = pd.DataFrame(data=iris.data, columns=iris.feature_names).to_numpy()
y_labels = iris.target
y_train = one_hot_encode(y_labels, num_classes=len(np.unique(y_labels)))
output_size = len(np.unique(y_labels))
hidden_layers = [128, 64]


input_size = X_train.shape[1]
architecture = [input_size] + hidden_layers + [output_size]

# Initialize the Neural Network model
model = NeuralNetwork(architecture)

class TrainRequest(BaseModel):
    alpha: float = 0.01
    epochs: int = 10

@app.get("/")
def home():
    return {"message": "Neural Network Visualization API is running!"}

@app.post("/train")
def train_model(train_request: TrainRequest):
    """
    Trains the model and returns loss/accuracy logs.
    """
    acc, losses = model.train(X_train, y_train, alpha=train_request.alpha, epochs=train_request.epochs)
    return {"accuracy": acc, "losses": losses}

@app.post("/predict")
def predict(data: list):
    """
    Performs a forward pass on given input data.
    """
    X_input = np.array(data).reshape(1, -1)
    predictions = model.forward_propagation(X_input)
    return {"prediction": predictions[f"A{len(model.weights)}"].tolist()}
