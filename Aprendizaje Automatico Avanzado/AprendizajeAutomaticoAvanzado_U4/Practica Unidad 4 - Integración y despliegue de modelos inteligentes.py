#!/usr/bin/env python
# coding: utf-8

# # Aprendizaje Automático Avanzado — Unidad 4
# ## Integración y Despliegue de Modelos Inteligentes
# 
# **Maestría en Inteligencia Artificial · TecNM Virtual**
# 
# Este cuaderno de prácticas acompaña la teoría de la Unidad 4. Contiene ejemplos **ejecutables** que ilustran los conceptos principales del tema, utilizando los conjuntos de datos de la asignatura.
# 
# > **Requisitos:** `pip install scikit-learn pandas numpy matplotlib`
# > Coloca los archivos `.csv` de la carpeta *Recursos y materiales* junto a este cuaderno (o ajusta la ruta `DATA`).
# 
# ---
# ### Contenido de la práctica
# 1. Construcción de un **pipeline** reproducible
# 2. Persistencia del modelo (`joblib`)
# 3. Función de **inferencia** lista para producción
# 4. Buenas prácticas de despliegue
# 

# ### 1. Pipeline reproducible
# Un `Pipeline` de scikit-learn encadena preprocesamiento y modelo en un solo objeto, evitando fugas de datos y facilitando el despliegue.

# In[1]:


import numpy as np, pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA = Path('.') if Path('breast_cancer_dataset.csv').exists() else Path('data')
df = pd.read_csv(DATA/'breast_cancer_dataset.csv')
X = df.drop(columns=['diagnosis'])
_d = df['diagnosis']; y = _d.map({'M':1,'B':0}) if _d.dtype==object else _d.astype(int)
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)

pipe = Pipeline([('scaler', StandardScaler()),
                 ('clf', LogisticRegression(max_iter=5000))]).fit(Xtr, ytr)
print('Exactitud del pipeline:', round(accuracy_score(yte, pipe.predict(Xte)),4))


# ### 2. Persistencia del modelo
# Guardamos el pipeline entrenado para reutilizarlo sin reentrenar.

# In[2]:


import joblib
joblib.dump(pipe, 'modelo_diagnostico.joblib')
modelo_cargado = joblib.load('modelo_diagnostico.joblib')
print('Modelo guardado y recargado correctamente:', type(modelo_cargado).__name__)


# ### 3. Función de inferencia
# Envolvemos la predicción en una función que recibiría datos nuevos (por ejemplo, desde una API).

# In[3]:


def predecir(muestra_dict):
    """Recibe un diccionario con las 30 características y devuelve el diagnóstico."""
    x = pd.DataFrame([muestra_dict])[X.columns]
    p = modelo_cargado.predict_proba(x)[0,1]
    return {'diagnostico': 'Maligno' if p>=0.5 else 'Benigno', 'probabilidad_maligno': round(float(p),3)}

ejemplo = X.iloc[0].to_dict()
print('Predicción de ejemplo:', predecir(ejemplo))


# ### 4. Buenas prácticas de despliegue
# - **Versionar** datos, código y modelo (MLOps).
# - **Monitorear** el desempeño y el *data drift* en producción.
# - Exponer el modelo mediante una **API** (FastAPI/Flask) o servicios en la nube.
# - Documentar entradas, salidas y limitaciones del modelo.
# 
# ---
# ### Ejercicio propuesto
# 1. Convierte la función `predecir` en un endpoint con **FastAPI**.
# 2. Añade validación de entradas (rangos plausibles de cada característica).
# 3. Investiga `MLflow` para el registro y seguimiento de experimentos.
