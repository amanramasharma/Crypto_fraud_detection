import torch
import pandas as pd
import numpy as np
import joblib
from app.model.autoencoder import AutoEncoder
from sklearn.preprocessing import MinMaxScaler,LabelEncoder

#loading the saved model components
MODEL_PATH = "app/model/autoencoder.pth"
SCALER_PATH = "app/model/scaler.pkl"
ENCODERS_PATH = "app/model/label_encoders.pkl"

def load_model(input_dim):
    model = AutoEncoder(input_dim)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model

def preprocess_input(new_data: pd.DataFrame):
    scaler: MinMaxScaler = joblib.load(SCALER_PATH)
    label_encoders: dict = joblib.load(ENCODERS_PATH)

    for col, encoder in label_encoders.items():
        if col in new_data.columns:
            new_data[col] = encoder.transform(new_data[col])

    new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
    new_data['timestamp'] = new_data['timestamp'].astype(int) / 10**9

    features = new_data.drop(columns=['user_id','wallet_id'], errors='ignore')
    features_scaled = scaler.transform(features)

    return torch.tensor(features_scaled, dtype=torch.float32)

def get_anomaly_score(model, input_tensor):
    with torch.no_grad():
        reconstruction = model(input_tensor)
        mse = torch.mean((input_tensor- reconstruction)**2, dim=1)
        return mse.numpy()
    
if __name__ == "__main__":
    sample_data = pd.read_csv("app/data/fraud_data.csv").head(5)
    model = load_model(input_dim=sample_data.shape[1]-2)
    processed = preprocess_input(sample_data)
    scores = get_anomaly_score(model,processed)
    for i, s in enumerate(scores):
        print(f"Sample {i+1}: Anomaly Score = {s:.4f}")