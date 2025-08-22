from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Security
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import mne
import tempfile
import io

# ----------------------------
# Load model
# ----------------------------
MODEL_PATH = "schizophrenia_detection_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ----------------------------
# API key (simple auth)
# ----------------------------
API_KEY = "my-secret-key"

def get_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI()

# Enable CORS (allow frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["https://lovable.ai"] in production
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
# Model info (so you know expected shape)
# ----------------------------
@app.get("/info")
def model_info():
    return {"expected_input_shape": model.input_shape}

# ----------------------------
# Helper functions for file preprocessing
# ----------------------------
def preprocess_csv(contents):
    # Read CSV from bytes and return as flat list
    import pandas as pd
    df = pd.read_csv(io.BytesIO(contents), header=None)
    # Flatten to 1D array (first channel, first 252 samples)
    eeg_data = df.values.flatten()[:252]
    return eeg_data

def preprocess_edf(contents):
    # Save to temp file and read with mne
    with tempfile.NamedTemporaryFile(delete=True, suffix=".edf") as tmp:
        tmp.write(contents)
        tmp.flush()
        raw = mne.io.read_raw_edf(tmp.name, preload=True)
        eeg_data = raw.get_data()[0][:252]  # First channel, first 252 samples
        return eeg_data

def preprocess_other(contents):
    raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a .csv or .edf file.")

# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    x_api_key: str = Security(get_api_key)
):
    try:
        contents = await file.read()
        filename = file.filename.lower()
        
        # Detect file type from extension and preprocess accordingly
        if filename.endswith(".csv"):
            eeg_data = preprocess_csv(contents)
        elif filename.endswith(".edf"):
            eeg_data = preprocess_edf(contents)
        else:
            eeg_data = preprocess_other(contents)

        eeg_array = np.array(eeg_data, dtype=float)

        expected_length = 252
        if eeg_array.shape[0] < expected_length:
            raise HTTPException(status_code=400, detail=f"Insufficient data length: expected at least {expected_length} values.")
        elif eeg_array.shape[0] > expected_length:
            eeg_array = eeg_array[:expected_length]  # truncate if longer

        input_data = eeg_array.reshape(1, expected_length, 1)

        y_pred = model.predict(input_data)
        predicted_class = int(np.argmax(y_pred))
        probabilities = y_pred.tolist()

        return {
            "filename": file.filename,
            "class": predicted_class,
            "probabilities": probabilities
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
