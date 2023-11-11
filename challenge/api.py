import fastapi
from fastapi import HTTPException
from pydantic import BaseModel
from challenge import DelayModel
import pickle
import re
from pathlib import Path


app = fastapi.FastAPI()
delay_model = DelayModel()  # Instantiate the model


BASE_DIR = Path(__file__).resolve(strict=True).parent


with open(f"{BASE_DIR}/reg_model.pkl", "rb") as f:
    model = pickle.load(f)

delay_model._model = model

class InputData(BaseModel):
    flights: list

@app.get("/health", status_code=200)
async def get_health() -> dict:
    if delay_model._model is not None:
        return {"status": "OK"}
    else:
        return {"status": "Model not loaded"}

@app.post("/predict", status_code=200)
async def post_predict(input_data: InputData) -> dict:
    if delay_model._model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Preprocess the input data
    input_features = delay_model.preprocess(input_data.flights)

    # Predict using the model
    predictions = delay_model.predict(input_features)

    # Return the predictions
    return {'predictions': predictions}