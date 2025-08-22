from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Security
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import mne
import tempfile
import io
import os
import pandas as pd
from supabase import create_client

# ----------------------------
# Load model
# ----------------------------
MODEL_PATH = "schizophrenia_detection_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ----------------------------
# Environment variables
# ----------------------------
API_KEY = os.getenv("API_KEY", "my-secret-key")
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://qaswbcayjdjrsjdmzpjv.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "your-service-role-key")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------------------
# API key validation
# ----------------------------
def get_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with ["https://lovable.ai"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Health check
# ----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running"}

# ----------------------------
# Model info
# ----------------------------
@app.get("/info")
def model_info():
    return {"expected_input_shape": model.input_shape}

# ----------------------------
# File preprocessing
# ----------------------------
def preprocess_csv(contents):
    df = pd.read_csv(io.BytesIO(contents), header=None)
    return df.values.flatten()[:252]

def preprocess_edf(contents):
    with tempfile.NamedTemporaryFile(delete=True, suffix=".edf") as tmp:
        tmp.write(contents)
        tmp.flush()
        raw = mne.io.read_raw_edf(tmp.name, preload=True)
        raw.pick_types(eeg=True)
        raw.crop(tmin=0, tmax=1)
        return raw.get_data()[0][:252]

def preprocess_other(contents):
    raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a .csv or .edf file.")

# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), x_api_key: str = Security(get_api_key)):
    try:
        contents = await file.read()
        filename = file.filename.lower()

        # Preprocess based on file type
        if filename.endswith(".csv"):
            eeg_data = preprocess_csv(contents)
        elif filename.endswith(".edf"):
            eeg_data = preprocess_edf(contents)
        else:
            eeg_data = preprocess_other(contents)

        eeg_array = np.array(eeg_data, dtype=float)

        # Validate and reshape
        expected_length = 252
        if eeg_array.shape[0] < expected_length:
            raise HTTPException(status_code=400, detail=f"Insufficient data length: expected at least {expected_length} values.")
        eeg_array = eeg_array[:expected_length]
        input_data = eeg_array.reshape(1, expected_length, 1)

        # Predict
        y_pred = model.predict(input_data)
        predicted_class = int(np.argmax(y_pred))
        probabilities = y_pred.tolist()

        # Log to Supabase
        try:
            supabase.table("predictions").insert({
                "filename": file.filename,
                "eeg_data": eeg_array.tolist(),
                "result_class": predicted_class,
                "probabilities": probabilities
            }).execute()
        except Exception as db_error:
            print("⚠️ Supabase insert failed:", db_error)

        return {
            "filename": file.filename,
            "class": predicted_class,
            "probabilities": probabilities
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
