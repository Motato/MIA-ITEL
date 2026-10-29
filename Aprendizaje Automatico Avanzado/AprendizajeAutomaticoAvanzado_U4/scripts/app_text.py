from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

# Cargar el modelo entrenado
MODEL_PATH = "modelo_ofensivo.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="API Detector de Lenguaje Ofensivo", version="1.0")

# Esquema de entrada
class TextPayload(BaseModel):
    text: str = Field(..., example="Eres un inútil")

@app.get("/")
def home():
    return {"status": "API activa", "endpoints": ["/docs", "/predict_text"]}

@app.post("/predict_text")
def predict_text(payload: TextPayload):
    text = payload.text.strip()
    if not text:
        return {"error": "El texto no puede estar vacío."}
    
    # Obtener probabilidades del modelo
    proba = model.predict_proba([text])[0]
    pred = int(np.argmax(proba))
    
    return {
        "label": "ofensivo" if pred == 1 else "no_ofensivo",
        "p_no_ofensivo": float(proba[0]),
        "p_ofensivo": float(proba[1]),
        "text": text
    }
    