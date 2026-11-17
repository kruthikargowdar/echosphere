from pathlib import Path
import joblib
import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

BACKEND_DIR = Path(__file__).resolve().parent
BASE_DIR = BACKEND_DIR.parent
MODEL_DIR = BASE_DIR if (BASE_DIR / "embedding_model").exists() else BASE_DIR / "echosphere_deployable"
EMBEDDING_MODEL_PATH = MODEL_DIR / "embedding_model"
CLASSIFIER_PATH = MODEL_DIR / "mlp_classifier.pkl"
EXPLAINER_PATH = MODEL_DIR / "shap_explainer.pkl"

def load_models():
    embedding_model = EncoderClassifier.from_hparams(
        source=str(EMBEDDING_MODEL_PATH), savedir=str(EMBEDDING_MODEL_PATH),
        run_opts={"device": "cpu"}
    )
    classifier = joblib.load(CLASSIFIER_PATH)
    explainer = joblib.load(EXPLAINER_PATH) if EXPLAINER_PATH.exists() else None
    return embedding_model, classifier, explainer

def extract_embedding(audio_file, embedding_model):
    waveform, sr = torchaudio.load(str(audio_file))
    if waveform.numel() == 0:
        raise ValueError("Audio file is empty")
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return embedding_model.encode_batch(waveform).squeeze().detach().cpu().numpy()

def predict_audio_file(audio_path, embedding_model, classifier, explainer):
    embedding = extract_embedding(audio_path, embedding_model)
    probabilities = classifier.predict_proba([embedding])[0]
    best_index = int(np.argmax(probabilities))
    class_value = int(classifier.classes_[best_index])
    if class_value not in (0, 1):
        raise ValueError(f"Unexpected classifier label: {class_value}")
    # The original project UI defines 1 as Real and 0 as Fake.
    label = "Real" if class_value == 1 else "Fake"
    confidence = float(probabilities[best_index]) * 100
    features = []
    if explainer is not None:
        try:
            explanation = explainer(embedding.reshape(1, -1))
            values = np.asarray(explanation.values)
            if values.ndim == 3:
                values = values[0, :, best_index]
            else:
                values = values.reshape(-1)
            for index in np.argsort(np.abs(values))[-8:][::-1]:
                features.append({"id": int(index), "description": "Learned embedding dimension (not a named acoustic feature)", "impact": float(values[index])})
        except Exception:
            features = []
    return label, confidence, features
