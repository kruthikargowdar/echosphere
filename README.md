# EchoSphere: Audio Deepfake Classification

EchoSphere runs a local audio classifier based on SpeechBrain ECAPA-TDNN embeddings and an MLP. The optional LLM feature explains the classifier output; it does not analyze the uploaded audio itself.

## Run locally

Use Python 3.11. From the `echosphere_deployable` directory, install `requirements.txt`, then run:

```powershell
python -m uvicorn backend.main:app --reload
```

The language model defaults to Gemini through Google's native `generateContent` API. Create a Gemini API key in Google AI Studio, then set it in PowerShell before starting the server:

```powershell
$env:GEMINI_API_KEY="your-gemini-api-key"
python -m uvicorn backend.main:app --reload
```

Gemini's free tier is subject to model availability and usage limits. Free-tier usage may be used by Google to improve its products; see the Gemini API terms before sending data. EchoSphere sends only the predicted label, confidence and numeric embedding signals to the LLM, never the audio. Without a key, audio classification works and the LLM explanation is disabled.

To use OpenAI instead, set `LLM_PROVIDER=openai` and `OPENAI_API_KEY`. You can set `GEMINI_MODEL` or `OPENAI_MODEL` to choose a model. `LLM_BASE_URL` can override the OpenAI-compatible endpoint when `LLM_PROVIDER=openai`.

PyTorch and torchaudio are pinned to matching CPU builds for compatibility with the bundled legacy SpeechBrain checkpoints.

## API and limits

- `GET /health` reports whether the model objects loaded.
- `POST /predict` accepts a WAV file in the multipart field `audio` (maximum 25 MB).
- `POST /explain` generates an optional text explanation.

Predictions are model outputs and should not be treated as proof. The embedding dimensions shown in the SHAP panel are learned numerical dimensions, not named acoustic features. The ZIP does not include reproducible training scripts, data splits, or independent evaluation results, so the previously reported validation accuracy cannot be independently verified.
