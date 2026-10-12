# ============================================================
# U1_supervisado_ejercicio_breastcancer.py
# Clasificación de Cáncer de Mama mediante Perceptrón Multicapa
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings

# Ignorar advertencias de convergencia si se prueba con pocas iteraciones
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 1) Ingesta de Datos
# ============================================================
# Se asume que 'diagnosis' es la variable objetivo (ej. M=Maligno, B=Benigno)
df = pd.read_csv("Aprendizaje Automatico Avanzado/data/breast_cancer_dataset.csv")

# Separación de variables predictoras (X) y la etiqueta (y)
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# ============================================================
# 2) Partición del Dataset
# ============================================================
# División 80/20 con semilla fija para garantizar la reproducibilidad
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 3) Preprocesamiento (Escalado de Características)
# ============================================================
# Crítico para redes neuronales: estandariza las características a media 0 y varianza 1.
# Evita que variables con magnitudes grandes dominen la función de pérdida.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Solo transform() para evitar fuga de datos

# ============================================================
# 4) Definición y Entrenamiento del Modelo Supervisado
# ============================================================
# MLP con una capa oculta de 50 neuronas. 
# max_iter se incrementa a 1000 para asegurar convergencia con datos complejos.
clf = MLPClassifier(
    hidden_layer_sizes=(50,), 
    max_iter=1000, 
    random_state=42,
    early_stopping=True # Previene el sobreajuste (overfitting)
)
clf.fit(X_train_scaled, y_train)

# ============================================================
# 5) Evaluación Clínica del Modelo
# ============================================================
y_pred = clf.predict(X_test_scaled)

print("\n" + "="*50)
print(" REPORTE DE CLASIFICACIÓN ONCOLÓGICA")
print("="*50 + "\n")

# classification_report expone Sensibilidad (Recall), crucial en diagnósticos médicos
print(classification_report(y_test, y_pred))

print("\nMATRIZ DE CONFUSIÓN:")
print("-" * 20)
print(confusion_matrix(y_test, y_pred))
print("-" * 20)