import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pickle

st.set_page_config(
    page_title="Student Pass/Fail Predictor",
    page_icon="PF",
    layout="centered",
)

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

# Custom styling
st.markdown(
    """
    <style>
        .main-card {
            background-color: #ffffff;
            padding: 1.5rem 1.75rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            border: 1px solid #f0f2f6;
        }
        .title-text {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .subtitle-text {
            font-size: 0.95rem;
            color: #5c6c80;
            margin-bottom: 1.25rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 999px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Layout
with st.sidebar:
    st.header("About this app")
    st.write(
        "This tool uses a simple neural network model to estimate "
        "whether a student is likely to pass or fail based on "
        "study habits and academic metrics."
    )
    st.markdown("**Tip:** Treat this as guidance, not a final decision.")

# st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("<div class='title-text'>Student Pass/Fail Predictor</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle-text'>Enter the student's details below to estimate the probability of passing the course.</div>",
    unsafe_allow_html=True,
)

with st.form("prediction_form"):
    st.subheader("Student details")

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.number_input(
            "Study hours per week",
            min_value=0,
            max_value=20,
            value=10,
            help="Approximate average number of hours spent studying each week.",
        )
        attendance = st.number_input(
            "Attendance (%)",
            min_value=0,
            max_value=100,
            value=75,
            help="Overall class attendance percentage.",
        )
        previous_grade = st.number_input(
            "Previous grade (0–100)",
            min_value=0,
            max_value=100,
            value=70,
            help="Score from the previous exam or course.",
        )

    with col2:
        assignments_completed = st.number_input(
            "Assignments completed",
            min_value=0,
            max_value=10,
            value=6,
            help="How many assignments have been completed.",
        )
        participation = st.number_input(
            "Class participation (0–10)",
            min_value=0,
            max_value=10,
            value=5,
            help="Overall engagement and interaction level in class.",
        )

    submitted = st.form_submit_button("Predict performance")

if submitted:
    # Prepare input
    X_new = np.array(
        [[study_hours, attendance, assignments_completed, previous_grade, participation]]
    )
    X_scaled = scaler.transform(X_new)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        prob = model(X_tensor).item()
        pred_class = "Pass" if prob > 0.5 else "Fail"

    prob_percent = prob * 100

    col_result_1, col_result_2 = st.columns(2)
    with col_result_1:
        st.metric("Pass probability", f"{prob_percent:.2f}%")
    with col_result_2:
        st.metric("Prediction", pred_class)

    st.progress(int(prob_percent))

    if pred_class == "Pass":
        st.success("The model suggests this student is likely to pass.")
    else:
        st.error("The model suggests this student is at risk of failing.")

    st.caption(
        "This is a model-based estimate and should be combined with "
        "teacher judgement and other qualitative information."
    )

st.markdown("</div>", unsafe_allow_html=True)
