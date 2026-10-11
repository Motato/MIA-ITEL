# ============================================================
# Librerías necesarias
# ============================================================

import time                                # Medir tiempo de ejecución
import warnings                            # Ignorar advertencias
import pandas as pd                        # Manejo de datasets (dataframes)
import seaborn as sns                      # Visualización estadística
import matplotlib.pyplot as plt            # Gráficos (matriz de confusión)

# Scikit-learn (Machine Learning)
from sklearn.model_selection import train_test_split   # División entrenamiento/prueba
from sklearn.neural_network import MLPClassifier       # Red Neuronal Perceptrón Multicapa
from sklearn.metrics import confusion_matrix           # Matriz de confusión

# Configuración: ignorar warnings futuros
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# Encabezado del modelo
# ============================================================

print("\n" + "-" * 80)
print(" MODELO DE CLASIFICACIÓN CON UNA RED NEURONAL")
print("-" * 80 + "\n")
print(" DETECCIÓN DE VOCABULARIO DE CIBERSUICIDIO")

# ============================================================
# Carga del dataset
# ============================================================

inicio_tiempo = time.time()   # Inicia contador de tiempo

dataset = './data/dataset_cibersuicidio.csv'
df = pd.read_csv(dataset)

print("\nDimensión del Dataset:")
print(f"Registros: {df.shape[0]}")
print(f"Columnas:  {df.shape[1]}\n")

print("Variables predictoras (X):")
X = df[df.columns[:-1]].values   # Todas las columnas excepto la última
print(X)

print("\nVariable a predecir (Y):")
y = df[df.columns[-1]].values    # Última columna
print(y)

# Mostrar características utilizadas
print("\nCaracterísticas utilizadas en el análisis:")
print(df.keys())

# ============================================================
# División de datos en entrenamiento y prueba
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print("\nEntrenamiento:", len(X_train), "registros (70%)")
print("Pruebas:", len(X_test), "registros (30%)")

# ============================================================
# Definición y entrenamiento del modelo
# ============================================================

# Red Neuronal con 1 capa oculta de 4 neuronas y hasta 11,000 iteraciones
red = MLPClassifier(max_iter=11000, hidden_layer_sizes=(4))
red.fit(X_train, y_train)

print(f"\nAprendizaje del algoritmo en el entrenamiento: {red.score(X_test, y_test):.2%}\n")

# Modelo adicional para comparación
model = MLPClassifier()
print("Ajusta el modelo de datos al proceso de entrenamiento")
model.fit(X_train, y_train)

# ============================================================
# Matriz de confusión
# ============================================================

y_pred = red.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)

print("\nMATRIZ DE CONFUSIÓN")
sns.heatmap(cnf_matrix.T, square=True, annot=True, fmt='d', cbar=True)
plt.xlabel('Clase verdadera')
plt.ylabel('Clase predicha')
plt.title('Matriz de Confusión')
plt.show()

print("\nDescripción de la matriz:")
print("Verdaderos Negativos:", cnf_matrix[0, 0])
print("Verdaderos Positivos:", cnf_matrix[1, 1])
print("Falsos Positivos:", cnf_matrix[1, 0])
print("Falsos Negativos:", cnf_matrix[0, 1])

# ============================================================
# Resultados finales
# ============================================================

tiempo = time.time() - inicio_tiempo
precision = 1 - (y_test != y_pred).sum() / y_test.shape[0]

print("\nRESULTADOS DE CLASIFICACIÓN")
print("-" * 66)
print(f"Porcentaje de precisión: {precision * 100:.1f}%")
print(f"Errores de clasificación: {(y_test != y_pred).sum()} errores sobre {y_test.shape[0]} casos")
print(f"Tiempo de ejecución: {tiempo:.2f} segundos")
print("-" * 66)
