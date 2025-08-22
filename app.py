from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

# ----------------------------
# Load model
# ----------------------------
MODEL_PATH = "schizophrenia_detection_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ----------------------------
# API key (simple auth)
# ----------------------------
API_KEY = "my-secret-key"

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
# Request schema
# ----------------------------
class EEGRequest(BaseModel):
    eeg_data: list

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
# Prediction
# ----------------------------
@app.post("/predict")
def predict_eeg(request: EEGRequest, x_api_key: str = Header(None)):
    try:
        # --- API Key Check ---
        if x_api_key != API_KEY:
            raise HTTPException(status_code=403, detail="Invalid API Key")

        # --- Ensure numeric input ---
        try:
            eeg_data = np.array(request.eeg_data, dtype=float)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="All eeg_data values must be numbers (no strings)."
            )

        # Get expected input shape from model (e.g., (None, 252, 1))
        expected_shape = model.input_shape
        target_shape = expected_shape[1:]   # e.g., (252,1)

        # --- Flexible Input Handling ---
        if eeg_data.ndim == 1 and target_shape[1] == 1:
            # Case: flat list → reshape to (252,1)
            if eeg_data.shape[0] != target_shape[0]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Input length {eeg_data.shape[0]} does not match required {target_shape[0]}"
                )
            eeg_data = eeg_data.reshape(target_shape)

        elif eeg_data.shape != target_shape:
            raise HTTPException(
                status_code=400,
                detail=f"Input shape {eeg_data.shape} does not match required {target_shape}"
            )

        # Add batch dimension → (1, 252, 1)
        x = eeg_data.reshape(1, *target_shape)

        # --- Model Prediction ---
        y_pred = model.predict(x)

        return {
            "probabilities": y_pred.tolist(),
            "class": int(np.argmax(y_pred))
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
