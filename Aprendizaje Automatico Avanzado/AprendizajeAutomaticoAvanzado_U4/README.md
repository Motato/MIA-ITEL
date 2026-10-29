# Pipeline y Despliegue: Detección de Lenguaje Ofensivo

**Autor:** [Tu Nombre]
**Descripción:** Pipeline de Machine Learning (TF-IDF + Regresión Logística) para clasificar texto como ofensivo o no ofensivo, desplegado como una API REST usando FastAPI.

## Instrucciones de ejecución
1. Instalar dependencias: `pip install -r requirements.txt`
2. Entrenar el modelo: Entrar a la carpeta `scripts` y ejecutar `python train_text_model.py`. Esto generará el archivo `modelo_ofensivo.joblib`.
3. Levantar la API: Ejecutar `uvicorn app_text:app --reload --port 8001`
4. Probar la API: Abrir `http://127.0.0.1:8001/docs` en el navegador.
