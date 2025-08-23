import streamlit as st
import requests

st.title("🧠 Schizophrenia Detection from EEG")

# File uploader
uploaded_file = st.file_uploader("Upload EEG file (.csv or .edf)", type=["csv", "edf"])

# Prediction button
if uploaded_file is not None and st.button("Predict"):
    with st.spinner("Sending file to backend..."):
        url = "https://hackethon-production.up.railway.app/predict"
        headers = {"x-api-key": "my-secret-key"}
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}

        response = requests.post(url, headers=headers, files=files)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: Class {result['class']}")
            st.write("Probabilities:", result["probabilities"])
        else:
            st.error(f"Error: {response.text}")
