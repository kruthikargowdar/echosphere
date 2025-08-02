import os
import joblib
import shap
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

# ✅ Define correct paths here (relative to backend/)
EMBEDDING_MODEL_PATH = "../embedding_model"
CLASSIFIER_PATH = "../mlp_classifier.pkl"
EXPLAINER_PATH = "../shap_explainer.pkl"

def load_models():
    print("🔁 Loading models...")

    embedding_model = EncoderClassifier.from_hparams(
        source=EMBEDDING_MODEL_PATH,
        savedir=EMBEDDING_MODEL_PATH,
        run_opts={"device": "cpu"}
    )

    with open(CLASSIFIER_PATH, "rb") as f:
        classifier = joblib.load(f)

    with open(EXPLAINER_PATH, "rb") as f:
        explainer = joblib.load(f)

    print("✅ Models loaded successfully!")
    return embedding_model, classifier, explainer

    return embedding_model, classifier, explainer


def extract_embedding(audio_file, embedding_model):
    waveform, sr = torchaudio.load(audio_file)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    embeddings = embedding_model.encode_batch(waveform)
    return embeddings.squeeze().detach().numpy()


def predict_audio_file(audio_path, embedding_model, classifier, explainer):
    embedding = extract_embedding(audio_path, embedding_model)
    proba = classifier.predict_proba([embedding])[0]
    label = classifier.classes_[np.argmax(proba)]
    confidence = max(proba)

    # SHAP explainability
    shap_values = explainer(embedding.reshape(1, -1))
    return label, confidence, shap_values
