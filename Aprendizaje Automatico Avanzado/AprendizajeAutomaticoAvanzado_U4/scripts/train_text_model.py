import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Dataset pequeño incrustado para ejecución rápida
texts = [
    "Eres un inútil", "Hola, ¿cómo estás?", "Qué asco de persona", 
    "Me gusta mucho", "Te odio", "Buen trabajo", 
    "Eres estúpido", "Felicidades", "Maldito seas", "Excelente día"
]
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] # 1 = Ofensivo, 0 = No ofensivo

MODEL_PATH = "modelo_ofensivo.joblib"

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000))
])

print("Entrenando el modelo...")
pipeline.fit(texts, labels)
joblib.dump(pipeline, MODEL_PATH)
print("[OK] Modelo guardado en:", MODEL_PATH)

