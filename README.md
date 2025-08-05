🎧 EchoSphere: Audio Deepfake Detection
A real-time AI-powered system to detect AI-generated audio using ECAPA-TDNN + MLP Classifier + SHAP Explainability.

💡 Features
Detects Real vs Fake (AI-generated) voices
Built with SpeechBrain, FastAPI, and MLP Classifier
SHAP-based explainability
Clean and aesthetic frontend (HTML/CSS)
Modular project structure
🛠️ Tech Stack
Python, FastAPI, HTML/CSS
SpeechBrain ECAPA-TDNN
SHAP for feature explanation
GitHub for version control
🧠 Model Training
Trained on WaveFake Dataset (2s clips)
Achieved ~99% accuracy on validation
🚀 Getting Started
cd echosphere_deployable/backend
uvicorn main:app --reload