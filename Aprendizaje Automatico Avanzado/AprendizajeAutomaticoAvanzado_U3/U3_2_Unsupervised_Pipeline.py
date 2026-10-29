import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def run_unsupervised_pipeline():
    # 1. Crear carpeta de resultados
    os.makedirs('resultados', exist_ok=True)
    print("Descargando y preparando Fashion-MNIST...")

    # 2. Cargar y Preprocesar (Normalización implícita al dividir por 255)
    dataset = datasets.FashionMNIST(root='./data', train=True, download=True)
    
    # Aplanar las imágenes de 28x28 a vectores de 784 dimensiones
    X = dataset.data.numpy().reshape(-1, 784) / 255.0
    y_true = dataset.targets.numpy()

    # 3. Reducción de Dimensionalidad (PCA: 784 -> 50 -> 2)
    print("Aplicando PCA para reducir de 784 a 50 componentes...")
    pca_50 = PCA(n_components=50, random_state=42)
    X_50 = pca_50.fit_transform(X)
    
    print("Aplicando PCA para reducir de 50 a 2 componentes para visualización...")
    pca_2 = PCA(n_components=2, random_state=42)
    X_2 = pca_2.fit_transform(X_50)

    # 4. Agrupamiento con K-Means
    print("Ejecutando K-Means con 10 clústeres...")
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_2)

    # 5. Visualización y Guardado
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_2[:, 0], X_2[:, 1], c=clusters, cmap='tab10', s=2, alpha=0.6)
    plt.colorbar(scatter, label='ID del Clúster')
    plt.title('Clústeres K-Means en Espacio Latente PCA (2D) - Fashion MNIST')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig('resultados/pca_kmeans_clusters.png', dpi=300)
    print("Gráfica guardada en 'resultados/pca_kmeans_clusters.png'")
    plt.show()

if __name__ == "__main__":
    run_unsupervised_pipeline()
    
