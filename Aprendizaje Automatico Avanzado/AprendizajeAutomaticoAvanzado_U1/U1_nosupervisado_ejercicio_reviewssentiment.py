# ============================================================
# unsupervised/unsupervised.py
# Agrupamiento Semántico de Reseñas mediante NLP y LSA
# ============================================================

import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1) Ingesta de Datos
# ============================================================
df = pd.read_csv("./data/reviews_sentiment.csv", sep=";")

# ============================================================
# 2) Vectorización Semántica (TF-IDF)
# ============================================================
# Scikit-Learn NO soporta stop_words="spanish" de forma nativa (solo "english").
# Se debe importar una biblioteca léxica robusta como NLTK.
# nltk.download('stopwords') # Ejecutar solo una vez en el entorno

spanish_stopwords = stopwords.words('spanish')

# Limitamos a 1000 features para filtrar ruido léxico de baja frecuencia
vectorizer = TfidfVectorizer(stop_words=spanish_stopwords, max_features=1000)
X_sparse = vectorizer.fit_transform(df["Review Text"].astype(str))

# ============================================================
# 3) Reducción de Dimensionalidad (LSA)
# ============================================================
# CRÍTICO: PCA requiere X.toarray(), lo cual colapsa la memoria RAM en NLP.
# TruncatedSVD está diseñado matemáticamente para matrices dispersas (Sparse Matrices).
# Aplicado sobre TF-IDF, este algoritmo se conoce como Latent Semantic Analysis (LSA).
svd = TruncatedSVD(n_components=2, random_state=42)
X_reduced = svd.fit_transform(X_sparse)

# ============================================================
# 4) Clustering No Supervisado
# ============================================================
# Agrupación espacial basada en las coordenadas semánticas reducidas
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_reduced)
df["Cluster"] = clusters

# ============================================================
# 5) Evaluación Matemática del Agrupamiento
# ============================================================
# El Coeficiente de Silueta valida empíricamente si k=3 es óptimo
score = silhouette_score(X_reduced, clusters)

print("\n" + "="*50)
print(" ANÁLISIS DE RAZONAMIENTO SEMÁNTICO (CLUSTERING)")
print("="*50)
print(f"Coeficiente de Silueta: {score:.4f} (Rango de -1 a 1)")
print("\nMuestra de la Asignación Semántica:")
print(df[["Review Title", "Cluster"]].head())