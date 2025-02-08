from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from NeuralNetwork import NeuralNetwork
from datasets import load_dataset  # Assume a function to load the dataset
import uuid
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to allow specific domains (e.g., ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods like GET, POST, OPTIONS
    allow_headers=["*"],  # Allow all headers
)

user_sessions = {}

@app.get("/")
def home():
    return {"message": "Neural Network Visualization API is running!"}

# ------------------ Model Initialization Request ------------------ #
class InitModelRequest(BaseModel):
    layer_sizes: list[int]  # List of layer sizes, excluding input & output
    activations: list[str]  # Activation function for each layer (except input)
    dataset: str  # Name of dataset (e.g., "california_housing", "mnist")

@app.post("/init_model")
def init_model(request: InitModelRequest):
    # Generate a unique session ID for the user
    session_id = str(uuid.uuid4())

    # Load dataset
    X_train, _, Y_train, _, input_size, output_size = load_dataset(request.dataset)

    layers = [input_size] + request.layer_sizes + [output_size]

    # Ensure activations length matches hidden + output layers
    if len(request.activations) != len(request.layer_sizes) + 1:
        return {"error": "Activations length must match number of layers - 1"}

    # Initialize neural network
    network = NeuralNetwork(layers, request.activations)

    # Store the model and dataset in the user's session
    user_sessions[session_id] = {
        "network": network,
        "X_train": X_train,
        "Y_train": Y_train
    }

    return {
        "message": "Model initialized successfully",
        "session_id": session_id,
        "layer_sizes": layers,
        "network" : network.to_dict()
    }


# ------------------ Training Request ------------------ #
class TrainRequest(BaseModel):
    session_id: str  # User's session ID
    learning_rate: float = 0.01
    epochs: int = 10

@app.post("/train")
def train_model(request: TrainRequest):
    # Retrieve the user's session
    session = user_sessions.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /init_model first.")

    network = session["network"]
    X_train = session["X_train"]
    Y_train = session["Y_train"]

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

# ------------------ Clear Session ------------------ #
@app.post("/clear_session")
def clear_session(session_id: str):
    if session_id in user_sessions:
        del user_sessions[session_id]
        return {"message": "Session cleared successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found.")