from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Literal, Annotated
import joblib
import pandas as pd

model = joblib.load("model_1.joblib")

# Vectorizer load
cv = joblib.load("vectorizer_1.joblib")

app = FastAPI()

# Home route
@app.get("/")
def home():
    return {"message": "News Bias Detection API is running 🚀"}

class NewsInput(BaseModel):
    text: str

@app.post("/predict/")
def predict(data: NewsInput):
    # Text को vectorize करना
    vectorized_text = cv.transform([data.text])

    # Prediction करना
    prediction = model.predict(vectorized_text)[0]

    # Mapping 0 → Un-Biased, 1 → Biased
    label = "Un-Biased" if prediction == 0 else "Biased"

    return {
        "input_text": data.text, 
        "bias_prediction": int(prediction),
        "label":label
        }
