import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the common model
model = joblib.load("common_model.pkl")

st.title("AI Application with Common Model")

# Input Section
st.header("Enter Feature Values")
user_input = st.text_area("Enter comma-separated feature values:")

if user_input:
    try:
        features = np.array([float(i) for i in user_input.split(",")]).reshape(1, -1)
        prediction = model.predict(features)
        st.success(f"Prediction: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Invalid input! Error: {e}")

# Batch Prediction
st.header("Batch Predictions")
uploaded_file = st.file_uploader("Upload a CSV file for batch predictions:", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        predictions = model.predict(data)
        st.write("Predictions:")
        st.write(pd.DataFrame(predictions, columns=["Prediction"]))
    except Exception as e:
        st.error(f"Error processing file: {e}")
