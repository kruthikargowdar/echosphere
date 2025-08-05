from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path

# Import your model utils
from .model_utils import load_models, predict_audio_file

app = FastAPI()

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve frontend index.html
@app.get("/", response_class=FileResponse)
def read_index():
    return FileResponse("static/index.html")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
embedding_model, classifier, explainer = load_models()

# Predict endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        temp_file_path = "temp_audio.wav"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())

        label, confidence, shap_values = predict_audio_file(
            temp_file_path, embedding_model, classifier, explainer
        )

        return {
            "prediction": label,
            "confidence": confidence,
            "shap_features": shap_values
        }

    except Exception as e:
        return {"error": str(e)}
