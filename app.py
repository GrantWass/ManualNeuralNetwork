from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from NeuralNetwork import NeuralNetwork
from datasets import load_dataset  # Assume a function to load the dataset
import uuid
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Union

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to allow specific domains (e.g., ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods like GET, POST, OPTIONS
    allow_headers=["*"],  # Allow all headers
)

user_sessions = {}

# ------------------ Model Initialization Request ------------------ #
class InitModelRequest(BaseModel):
    layer_sizes: List[int]  # List of layer sizes, excluding input & output
    activations: List[str]  # Activation function for each layer (except input)
    dataset: str  # Name of dataset (e.g., "california_housing", "mnist")

class InitModelResponse(BaseModel):
    message: str
    session_id: str
    layer_sizes: List[int]
    original_train_data: list
    network: dict  # Serialized network as dictionary

@app.post("/init_model", response_model=InitModelResponse)
def init_model(request: InitModelRequest):
    session_id = str(uuid.uuid4())  # Generate a unique session ID

    # Load dataset
    X_train, _, Y_train, _, input_size, output_size, output_activation, original_train_data = load_dataset(request.dataset)

    layers = [input_size] + request.layer_sizes + [output_size]

    # Ensure activations length matches hidden + output layers
    if len(request.activations) != len(request.layer_sizes):
        raise HTTPException(status_code=400, detail="Activations length must match number of layers.")

    activations = request.activations + [output_activation]
    network = NeuralNetwork(layers, activations)

    # Store the model and dataset in the user's session
    user_sessions[session_id] = {
        "network": network,
        "X_train": X_train,
        "Y_train": Y_train,
    }

    return InitModelResponse(
        message="Model initialized successfully",
        session_id=session_id,
        layer_sizes=layers,
        original_train_data=original_train_data,
        network=network.to_dict()  # Return serialized network
    )


# ------------------ Training Request ------------------ #
class TrainRequest(BaseModel):
    session_id: str  # User's session ID
    learning_rate: float = 0.01
    epochs: int = 10

class LayerDetail(BaseModel):
    weights: list
    biases: list
    Z: list
    A: list
    dW: list
    db: list
    dZ: list
    activation: str

class TrainResult(BaseModel):
    epoch: int
    input: list
    loss: float
    name: str  # Metric name (e.g., accuracy, mae)
    metric: Union[float, str]  # Metric could be accuracy or mae
    layers: List[LayerDetail]

class TrainResponse(BaseModel):
    training_results: List[TrainResult]

@app.post("/train", response_model=TrainResponse)
def train_model(request: TrainRequest):
    session = user_sessions.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /init_model first.")

    network = session["network"]
    X_train = session["X_train"]
    Y_train = session["Y_train"]

    training_results = []

    for epoch in range(request.epochs):
        result = network.train_step(X_train, Y_train, request.learning_rate) # TODO FIX LOSS FOR REGRESSION

        layers = [
            LayerDetail(
                weights=layer["weights"].tolist(),
                biases=layer["biases"].tolist(),
                Z=layer["Z"].tolist(),
                A=layer["A"].tolist(),
                dW=layer["dW"].tolist(),
                db=layer["db"].tolist(),
                dZ=layer["dZ"].tolist(),
                activation=layer["activation"]
            ) for layer in result["layers"]
        ]

        metric_name = "accuracy" if "accuracy" in result else "mae"
        metric_value = result.get("accuracy") if "accuracy" in result else result.get("mae")

        # Add epoch results
        training_results.append(TrainResult(
            epoch=epoch + 1,
            input= X_train.tolist(),
            loss=result["loss"],
            name=metric_name,
            metric=metric_value,
            layers=layers
        ))

    return TrainResponse(training_results=training_results)


# ------------------ Clear Session ------------------ #
@app.post("/clear_session")
def clear_session(session_id: str):
    if session_id in user_sessions:
        del user_sessions[session_id]
        return {"message": "Session cleared successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found.")
