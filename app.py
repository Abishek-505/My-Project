import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pickle

# Load model & scaler
with open("pass_fail_model.pkl", "rb") as f:
    data = pickle.load(f)

scaler = data['scaler']
input_dim = data['input_dim']

# Define same model structure
class SimpleANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = SimpleANN(input_dim=input_dim)
model.load_state_dict(data['model_state'])
model.eval()

# Streamlit UI
st.title("Student Pass/Fail Predictor")

study_hours = st.number_input("Study Hours per Week", min_value=0, max_value=20, value=10)
attendance = st.number_input("Attendance %", min_value=0, max_value=100, value=75)
assignments_completed = st.number_input("Assignments Completed", min_value=0, max_value=10, value=6)
previous_grade = st.number_input("Previous Grade (0-100)", min_value=0, max_value=100, value=70)
participation = st.number_input("Class Participation (0-10)", min_value=0, max_value=10, value=5)

if st.button("Predict"):
    # Prepare input
    X_new = np.array([[study_hours, attendance, assignments_completed, previous_grade, participation]])
    X_scaled = scaler.transform(X_new)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # Predict
    with torch.no_grad():
        prob = model(X_tensor).item()
        pred_class = "Pass" if prob > 0.5 else "Fail"
    
    st.success(f"Predicted Pass Probability: {prob*100:.2f}%")
    st.info(f"Predicted Class: {pred_class}")
