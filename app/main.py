from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import torch
import joblib
from app.model.autoencoder import AutoEncoder
from app.model.predict import preprocess_input, get_anomaly_score

app = FastAPI(title="Crypto Fraud Detection API")

MODEL_PATH = "app/model/autoencoder.pth"
SCALER_PATH = "app/model/scaler.pkl"
ENCODERS_PATH = "app/model/label_encoders.pkl"


sample_data = pd.read_csv("app/data/fraud_data.csv")
input_dim = sample_data.drop(columns=["user_id", "wallet_id"]).shape[1]


model = AutoEncoder(input_dim)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

class Transaction(BaseModel):
    user_id: str
    wallet_id: str
    ip_address: str
    amount: float
    tx_type: str
    timestamp: str
    location: str
    device: str
    is_night: int

@app.get("/")
def root():
    return {"message": "Crypto Fraud Detection API is live."}

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([transaction.dict()])
        
        # Preprocess and predict
        processed = preprocess_input(input_df)
        score = get_anomaly_score(model, processed)[0]

        return {
            "anomaly_score": round(float(score), 4),
            "is_fraud": score > 0.02
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))