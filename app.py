from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from NeuralNetwork import NeuralNetwork
from datasets import load_dataset  # Assume a function to load the dataset

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Neural Network Visualization API is running!"}

# Global variables
network = None
X_train, Y_train = None, None  # Will store dataset

# ------------------ Model Initialization Request ------------------ #
class InitModelRequest(BaseModel):
    layer_sizes: list[int]  # List of layer sizes, including input & output
    activations: list[str]  # Activation function for each layer (except input)
    dataset: str  # Name of dataset (e.g., "california_housing", "mnist")

@app.post("/init_model")
def init_model(request: InitModelRequest):
    global network, X_train, Y_train

    # Load dataset
    X_train, _, Y_train, _, input_size, output_size = load_dataset(request.dataset)

    # Ensure first and last layers match dataset dimensions
    if request.layer_sizes[0] != input_size or request.layer_sizes[-1] != output_size:
        return {"error": "First layer size must match input size, and last layer must match output size"}

    # Ensure activations length matches hidden + output layers
    if len(request.activations) != len(request.layer_sizes) - 1:
        return {"error": "Activations length must match number of layers - 1"}

    # Initialize neural network
    network = NeuralNetwork(request.layer_sizes, request.activations)
    
    return {"message": "Model initialized successfully", "layer_sizes": request.layer_sizes}


# ------------------ Training Request ------------------ #
class TrainRequest(BaseModel):
    learning_rate: float = 0.01
    epochs: int = 10

@app.post("/train")
def train_model(request: TrainRequest):
    global network, X_train, Y_train
    if network is None:
        return {"error": "Model not initialized. Call /init_model first."}

    training_results = []

    for epoch in range(request.epochs):
        result = network.train_step(X_train, Y_train, request.learning_rate)
        training_results.append({
            "epoch": epoch + 1,
            "loss": result["loss"],
            "metric": result.get("accuracy", result.get("mae")),
            "layers": result["layers"]
        })

    return {"training_results": training_results}

