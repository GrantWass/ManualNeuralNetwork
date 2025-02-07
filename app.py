from fastapi import FastAPI
from pydantic import BaseModel
from NeuralNetwork import NeuralNetwork

app = FastAPI()

class TrainRequest(BaseModel):
    alpha: float = 0.01
    epochs: int = 10

@app.get("/")
def home():
    return {"message": "Neural Network Visualization API is running!"}

@app.post("/train")
def train_model(train_request: TrainRequest):
    return 
