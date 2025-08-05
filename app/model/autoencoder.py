import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
import os

class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim,64), nn.ReLU(),
                                     nn.Linear(64,32), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(32,64),nn.ReLU(),nn.Linear(64,input_dim))

    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
def train_autoencoder(data_path, model_path, scaler_path,encoders_path):
    df = pd.read_csv(data_path)

    categorical_cols = ['ip_address', 'tx_type', 'location', 'device']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = df['timestamp'].astype(int) / 10**9

    features = df.drop(columns=['user_id','wallet_id'])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features)

    X_tensor = torch.tensor(X_scaled,dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset,batch_size=128,shuffle=True)

    model = AutoEncoder(input_dim=X_scaled.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    for epoch in range(10):
        total_loss = 0
        for batch in loader:
            x = batch[0]
            output = model(x)
            loss = criterion(output,x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

    torch.save(model.state_dict(),model_path)
    joblib.dump(scaler,scaler_path)
    joblib.dump(label_encoders, encoders_path)

if __name__ == "__main__":
    os.makedirs("app/model", exist_ok=True)
    train_autoencoder(data_path="app/data/fraud_data.csv",model_path="app/model/autoencoder.pth",
                      scaler_path = "app/model/scaler.pkl", encoders_path="app/model/label_encoders.pkl")