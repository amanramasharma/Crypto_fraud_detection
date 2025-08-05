import shap
import torch
import pandas as pd
import joblib
from app.model.autoencoder import AutoEncoder
from app.model.predict import preprocess_input

# Load model & tools
MODEL_PATH = "app/model/autoencoder.pth"
SCALER_PATH = "app/model/scaler.pkl"
ENCODERS_PATH = "app/model/label_encoders.pkl"
sample_data = pd.read_csv("app/data/fraud_data.csv")

input_dim = sample_data.drop(columns=["user_id", "wallet_id"]).shape[1]
model = AutoEncoder(input_dim)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Preprocess a sample batch
X = preprocess_input(sample_data.head(50))

# Use KernelExplainer since we have no gradients
background = X[:20].numpy()
explainer = shap.KernelExplainer(model.forward, background)
shap_values = explainer.shap_values(X[:5].numpy())

# Plot SHAP for one prediction
shap.initjs()
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    sample_data.drop(columns=["user_id", "wallet_id"]).iloc[0],
    matplotlib=True
)