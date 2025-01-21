import pandas as pd
import shap
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import h2o
from h2o.automl import H2OAutoML
import hashlib
from web3 import Web3

# Streamlit App Title
st.title("AI-Powered Explainable Business System with Blockchain Integration")

# Upload File
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.write(data.head())

    # Automated Data Profiling
    if st.button("Generate Data Profiling Report"):
        from pandas_profiling import ProfileReport
        profile = ProfileReport(data, title="Data Profiling Report", explorative=True)
        profile.to_file("data_profile.html")
        st.success("Data Profiling Report Generated: data_profile.html")

    # Preprocessing
    target_col = st.text_input("Enter Target Column Name:")
    if target_col in data.columns:
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = LabelEncoder().fit_transform(data[col])
        X = data.drop(columns=[target_col])
        y = data[target_col]

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

  # Install Auto-sklearn
!pip install auto-sklearn

import autosklearn.classification
from sklearn.metrics import accuracy_score

# Replace H2O AutoML with Auto-sklearn
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=300, per_run_time_limit=30)
automl.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = automl.predict(X_val)
st.write("AutoML Best Model Accuracy:", accuracy_score(y_val, y_pred))


        # Random Forest Model
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        st.write("Validation Performance:")
        st.text(classification_report(y_val, y_pred))

        # SHAP Explainability
        if st.button("Generate SHAP Explanations"):
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_val)
            st.subheader("Global Explanation")
            shap.summary_plot(shap_values[1], X_val, feature_names=X.columns, show=False)
            st.pyplot(plt)

            st.subheader("Local Explanation")
            instance_idx = st.slider("Select Instance", 0, len(X_val) - 1, 0)
            shap.force_plot(explainer.expected_value[1], shap_values[1][instance_idx], X_val[instance_idx], matplotlib=True)
            st.pyplot(plt)

        # Advanced Plots
        if st.button("Show Advanced Plots"):
            importance = rf.feature_importances_
            fig = px.bar(x=X.columns, y=importance, labels={"x": "Features", "y": "Importance"}, title="Feature Importance")
            st.plotly_chart(fig)

            fig = px.scatter(x=y_val, y=y_pred, labels={"x": "Actual", "y": "Predicted"}, title="Predicted vs Actual")
            st.plotly_chart(fig)

        # Blockchain Integration
        if st.button("Log to Blockchain"):
            infura_url = "https://polygon-mumbai.infura.io/v3/YOUR_INFURA_PROJECT_ID"
            web3 = Web3(Web3.HTTPProvider(infura_url))

            contract_address = "0xYourContractAddress"
            abi = [...]  # Replace with your contract's ABI
            contract = web3.eth.contract(address=contract_address, abi=abi)

            def generate_hash(data):
                return hashlib.sha256(data.encode()).hexdigest()

            file_hash = generate_hash("Uploaded File")
            prediction_hash = generate_hash(str(y_pred.tolist()))
            explanation_hash = generate_hash("SHAP Explanation")

            tx = contract.functions.addRecord(file_hash, prediction_hash, explanation_hash).buildTransaction({
                'from': web3.eth.default_account,
                'nonce': web3.eth.get_transaction_count(web3.eth.default_account),
            })
            private_key = "YOUR_PRIVATE_KEY"
            signed_tx = web3.eth.account.sign_transaction(tx, private_key=private_key)
            tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            st.write(f"Transaction Hash: {web3.toHex(tx_hash)}")

else:
    st.warning("Please upload a CSV file to proceed.")
