import json
import os
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from .model_utils import load_models, predict_audio_file
except ImportError:  # Supports `uvicorn main:app` from the backend directory.
    from model_utils import load_models, predict_audio_file

BASE_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
app = FastAPI(title="EchoSphere")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
embedding_model, classifier, explainer = load_models()

@app.get("/")
def read_index():
    return FileResponse(BASE_DIR / "static" / "index.html")

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": embedding_model is not None and classifier is not None}

@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    if not audio.filename or Path(audio.filename).suffix.lower() != ".wav":
        raise HTTPException(status_code=415, detail="Please upload a WAV audio file.")
    content = await audio.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio file exceeds the 25 MB limit.")
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
            temporary.write(content)
            temp_path = temporary.name
        label, confidence, features = predict_audio_file(temp_path, embedding_model, classifier, explainer)
        return {"prediction": label, "confidence": confidence, "shap_features": features}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process this WAV file: {exc}") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

class ExplainRequest(BaseModel):
    prediction: str
    confidence: float
    features: list[dict] = Field(default_factory=list)

@app.post("/explain")
def explain(request: ExplainRequest):
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        key_name = "GEMINI_API_KEY"
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        endpoint = f"{base_url}/chat/completions"
        key_name = "OPENAI_API_KEY"
    else:
        raise HTTPException(status_code=400, detail="LLM_PROVIDER must be 'gemini' or 'openai'.")

    if not api_key:
        raise HTTPException(status_code=503, detail=f"LLM explanations are not configured. Set {key_name} on the server.")

    system_prompt = "Explain this audio classifier result in plain language in 2-3 short sentences. Treat the classifier output as uncertain evidence, never as proof. Do not invent audio observations or claim embedding dimensions are acoustic features. Mention that the model has not been independently validated."
    evidence = {"prediction": request.prediction, "confidence_percent": request.confidence, "top_embedding_dimensions": request.features[:8]}
    if provider == "gemini":
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": json.dumps(evidence)}]},],
        }
        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    else:
        payload = {"model": model, "temperature": 0.2, "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(evidence)},
        ]}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    req = Request(endpoint, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urlopen(req, timeout=30) as response:
            result = json.loads(response.read())
        if provider == "gemini":
            candidates = result.get("candidates", [])
            parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
            explanation = " ".join(part.get("text", "") for part in parts if part.get("text"))
        else:
            explanation = result["choices"][0]["message"]["content"]
        if not explanation:
            raise HTTPException(status_code=502, detail=f"{provider.title()} returned no explanation text.")
        return {"explanation": explanation, "model": model, "provider": provider}
    except HTTPException:
        raise
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            provider_error = json.loads(body).get("error", {}).get("message", body)
        except (ValueError, AttributeError):
            provider_error = body
        safe_detail = str(provider_error).replace(api_key, "[redacted]")[:400]
        raise HTTPException(status_code=502, detail=f"{provider.title()} API error (HTTP {exc.code}): {safe_detail}") from exc
    except (URLError, TimeoutError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"Could not get an explanation from {provider}: {type(exc).__name__}.") from exc
