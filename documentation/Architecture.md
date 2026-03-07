# 🫁 Proyecto: Detección de Neumonía en Rayos X con Explicabilidad (XAI)

## 📌 Contexto y Objetivo del Proyecto
Este proyecto NO es una herramienta médica para hospitales. Es una **demostración técnica de MLOps, Visión Computacional y Explicabilidad (XAI)** diseñada para un portfolio de Data Science / ML Engineering. 

El objetivo es demostrar la capacidad de tomar un dataset (Kaggle Chest X-Ray), resolver problemas de datos (desbalance, particiones incorrectas), entrenar un modelo pre-entrenado eficiente, y desplegarlo en producción utilizando buenas prácticas de ingeniería de software (Docker, APIs, UI interactiva).

## 🛠️ Stack Tecnológico Seleccionado
* **Entrenamiento (Cómputo Gratuito):** Kaggle Notebooks o Google Colab (Uso de GPU para el entrenamiento).
* **Machine Learning (Core):** PyTorch.
* **Arquitectura del Modelo:** ResNet18 (Ligero, rápido e ideal para inferencia en CPU sin costes).
* **Técnica de Explicabilidad (XAI):** Grad-CAM (Gradient-weighted Class Activation Mapping) para generar heatmaps que muestren en qué parte del pulmón se fija el modelo.
* **Backend / API:** FastAPI (Python) - Asíncrono y documentado automáticamente.
* **Frontend / UI:** Streamlit - Para una interfaz limpia donde el usuario pueda subir imágenes de prueba.
* **Despliegue e Infraestructura:** Docker + Hugging Face Spaces (Docker Space). Coste $0.



## 🔄 Flujo de Arquitectura (Core Loop)
1.  **Input:** El usuario entra a la web (Streamlit en Hugging Face) y sube una imagen de Rayos X (se proveerán ejemplos en la misma UI).
2.  **Petición:** Streamlit envía la imagen al endpoint de FastAPI.
3.  **Procesamiento (Inferencia):** FastAPI preprocesa la imagen, la pasa por el modelo PyTorch (.pth) cargado en memoria, y obtiene la predicción.
4.  **Explicabilidad:** Se ejecuta el algoritmo Grad-CAM sobre la última capa convolucional de ResNet18 para generar el heatmap.
5.  **Output:** FastAPI devuelve un JSON con la predicción (ej. `{"clase": "Pneumonia", "confianza": 0.98}`) y la imagen del heatmap codificada en base64.
6.  **Visualización:** Streamlit renderiza los resultados mostrando la imagen original junto al heatmap para demostrar "por qué" el modelo tomó esa decisión.

## 🗺️ Fases de Implementación (Hoja de Ruta para la IA)

### Fase 1: Datos y Entrenamiento (En Kaggle/Colab)
- Descargar dataset de Paul Mooney.
- **Crítico:** Arreglar el split de validación (el original solo tiene 16 imágenes). Unir `train` y `val` y hacer un nuevo split 80/20.
- Aplicar Data Augmentation básico (rotaciones ligeras) y pesos de clase (`class_weights`) para manejar el desbalance entre NORMAL y PNEUMONIA.
- Entrenar ResNet18 usando Transfer Learning. Optimizar para la métrica de **Recall**.
- Exportar el modelo entrenado (`modelo_resnet18.pth`).

### Fase 2: Desarrollo del Backend y Explicabilidad (Local)
- Crear el script `model_inference.py` que cargue el `.pth` y contenga la lógica de Grad-CAM.
- Crear `api.py` con FastAPI exponiendo el endpoint `/predict`.

### Fase 3: Desarrollo del Frontend (Local)
- Crear `app.py` con Streamlit.
- Diseñar la interfaz de subida de archivos y la llamada HTTP al backend.

### Fase 4: Contenerización y Despliegue (Local -> Hugging Face)
- Escribir un único `Dockerfile` que instale las dependencias (`requirements.txt`) y ejecute tanto FastAPI (uvicorn) como Streamlit en sus respectivos puertos, o unificarlos si es necesario para HF Spaces.
- Desplegar en Hugging Face Spaces usando la opción "Docker".