import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = load_model("credit_default_ann.h5")
scaler = StandardScaler()

# Define all 23 input features
features = ["LIMIT_BAL", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

def preprocess_input(data):
    """Compute averages and scale input data."""
    valid_pay_delays = [data[f'PAY_{i}'] for i in [0, 2, 3, 4, 5, 6] if data[f'BILL_AMT{i//2+1}'] != 0 or data[f'PAY_AMT{i//2+1}'] != 0]
    valid_bill_diffs = [data[f'BILL_AMT{i+1}'] - data[f'PAY_AMT{i+1}'] for i in range(6) if data[f'BILL_AMT{i+1}'] != 0 or data[f'PAY_AMT{i+1}'] != 0]
    
    data['AVG_PAY_DELAY'] = np.mean(valid_pay_delays) if valid_pay_delays else 0
    data['AVG_BILL_DIFF'] = np.mean(valid_bill_diffs) if valid_bill_diffs else 0
    
    processed_data = scaler.transform([[data[f] for f in ["LIMIT_BAL", "AVG_PAY_DELAY", "AVG_BILL_DIFF", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]]])
    return processed_data

def predict_single(input_data):
    """Make a single prediction."""
    processed_data = preprocess_input(input_data)
    prob = model.predict(processed_data)[0][0]
    return prob, "Default" if prob > 0.5 else "No Default"

def predict_batch(uploaded_file):
    """Predict on batch CSV file."""
    df = pd.read_csv(uploaded_file)
    df['AVG_PAY_DELAY'] = df.apply(lambda row: np.mean([row[f'PAY_{i}'] for i in [0, 2, 3, 4, 5, 6] if row[f'BILL_AMT{i//2+1}'] != 0 or row[f'PAY_AMT{i//2+1}'] != 0]), axis=1)
    df['AVG_BILL_DIFF'] = df.apply(lambda row: np.mean([row[f'BILL_AMT{i+1}'] - row[f'PAY_AMT{i+1}'] for i in range(6) if row[f'BILL_AMT{i+1}'] != 0 or row[f'PAY_AMT{i+1}'] != 0]), axis=1)
    
    df_processed = scaler.transform(df[["LIMIT_BAL", "AVG_PAY_DELAY", "AVG_BILL_DIFF", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]])
    predictions = model.predict(df_processed)
    df['Default Probability'] = predictions
    df['Prediction'] = df['Default Probability'].apply(lambda x: "Default" if x > 0.5 else "No Default")
    return df

# Streamlit UI
st.title("Credit Card Default Prediction")

# Single Input Prediction
st.header("Single Prediction")
input_values = {}
for feature in features:
    input_values[feature] = st.number_input(f"Enter {feature}", value=0.0)

if st.button("Predict Single Case"):
    prob, status = predict_single(input_values)
    st.write(f"Default Probability: {prob:.2f}")
    st.write(f"Prediction: {status}")

# Batch Prediction
st.header("Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    predictions_df = predict_batch(uploaded_file)
    st.write(predictions_df)
    st.download_button("Download Predictions", predictions_df.to_csv(index=False), "predictions.csv", "text/csv")
