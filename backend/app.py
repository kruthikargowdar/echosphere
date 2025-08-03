# echosphere_ui/assets/app.py

import os
import streamlit as st
import torch
import torchaudio
import numpy as np
import joblib
import streamlit.components.v1 as components
from speechbrain.inference.speaker import EncoderClassifier
import matplotlib.pyplot as plt


st.set_page_config(page_title="EchoSphere | Audio Deepfake Detector", layout="centered")
st.markdown("""
    <style>
        .stButton>button {
            background-color: #dab4ff;
            color: black;
            border-radius: 10px;
            padding: 0.5em 1em;
            font-weight: bold;
        }
        .stFileUploader { border: 2px dashed #dab4ff; padding: 1em; }
    </style>
""", unsafe_allow_html=True)

st.title("🎧 EchoSphere: Audio Deepfake Detection")
st.markdown("Upload a `.wav` audio file (16kHz, mono, ~2s) to check if it's **Real** or **AI-generated**.")

# ─── MODEL LOADERS ──────────────────────────────────
@st.cache_resource
def load_embedding_model():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "..", "echosphere_audio", "echosphere_models", "embedding_model")
    return EncoderClassifier.from_hparams(source=path, run_opts={"device": "cpu"})

@st.cache_resource
def load_classifier():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "..", "echosphere_audio", "echosphere_models", "mlp_classifier.pkl")
    return joblib.load(path)

@st.cache_resource
def load_shap_explainer():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "..", "echosphere_audio", "echosphere_models", "shap_explainer.pkl")
    return joblib.load(path)

embedding_model = load_embedding_model()
mlp_classifier  = load_classifier()
shap_explainer  = load_shap_explainer()

# ─── EMBEDDING EXTRACTION ───────────────────────────
def extract_embedding(wav, sr):
    if sr != 16000:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wav)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    emb = embedding_model.encode_batch(wav)
    return emb.squeeze().detach().cpu().numpy()

# ─── UPLOAD & PREDICT ─────────────────────────────────
uploaded = st.file_uploader("Upload WAV file", type="wav")
if uploaded:
    tmp = "temp_audio.wav"
    with open(tmp, "wb") as f:
        f.write(uploaded.read())

    try:
        wav, sr = torchaudio.load(tmp)
        st.audio(tmp, format="audio/wav")

        emb = extract_embedding(wav, sr).reshape(1, -1)
        pred = mlp_classifier.predict(emb)[0]
        proba= mlp_classifier.predict_proba(emb)[0][pred]

        label = "🟢 Real" if pred == 1 else "🔴 Fake"
        conf  = proba * 100

        st.markdown(f"### ✅ Prediction: **{label}**")
        st.markdown(f"**🧠 Confidence:** `{conf:.2f}%`")
        st.progress(conf / 100)

        # ─── SHAP EXPLANATION ─────────────────────────
        with st.expander("💡 SHAP Explanation"):
            sv = shap_explainer(emb)
            vals = sv.values[0]

            # Prepare top‑10 features
            top_idx = np.argsort(np.abs(vals))[::-1][:10]
            top_vals = vals[top_idx]
            top_names = [f"F{idx}" for idx in top_idx]

            # Draw horizontal bar chart
            fig, ax = plt.subplots(figsize=(6, 3))
            y_pos = np.arange(len(top_idx))
            ax.barh(y_pos, top_vals, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_names)
            ax.invert_yaxis()
            ax.set_xlabel("SHAP value")
            ax.set_title("Top 10 Feature Impacts")
            st.pyplot(fig)

            # List for clarity
            st.markdown("#### 🔎 Top 10 Features")
            for idx, val in zip(top_idx, top_vals):
                direction = "Fake" if val > 0 else "Real"
                emoji = "🔴" if val > 0 else "🟢"
                st.write(f"{emoji} Feature **{idx}** pushes towards **{direction}** by {val:.4f}")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
