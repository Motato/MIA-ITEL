import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ============================================================
# 1) Cargar dataset
# ============================================================
df = pd.read_csv("Aprendizaje Automatico Avanzado/data/dataset_cibersuicidio.csv")

# ============================================================
# 2) Variables predictoras (sin etiqueta)
# ============================================================
X = df.drop(columns=["Correcta"], errors="ignore")

# ============================================================
# 3) Escalamiento
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 4) PCA (2 componentes)
# ============================================================
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("Varianza explicada por PC1 y PC2:", pca.explained_variance_ratio_)
print("Varianza acumulada (2 componentes):", pca.explained_variance_ratio_.sum())

# ============================================================
# 5) K-means clustering
# ============================================================
k = 3  # número de clusters
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

cluster_counts = pd.Series(clusters).value_counts().sort_index()
print("\nTamaño por cluster:")
print(cluster_counts)

# ============================================================
# 6) Visualización PCA + KMeans
# ============================================================
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, s=15)
plt.title(f"PCA (2D) + K-means (k={k})")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.show()

# ============================================================
# 7) Interpretación de clusters
# ============================================================
X_with_cluster = X.copy()
X_with_cluster["cluster"] = clusters
cluster_means = X_with_cluster.groupby("cluster").mean(numeric_only=True)

top_n = 10
for c in range(k):
    top_features = cluster_means.loc[c].sort_values(ascending=False).head(top_n)
    print(f"\nTop {top_n} términos por frecuencia promedio en cluster {c}:")
    print(top_features)

# ============================================================
# 8) Comparación con etiqueta (opcional)
# ============================================================
if "Correcta" in df.columns:
    df_tmp = df.copy()
    df_tmp["cluster"] = clusters
    crosstab = pd.crosstab(df_tmp["cluster"], df_tmp["Correcta"], normalize="index")
    print("\nDistribución de la etiqueta 'Correcta' por cluster (solo interpretación):")
    print(crosstab)
