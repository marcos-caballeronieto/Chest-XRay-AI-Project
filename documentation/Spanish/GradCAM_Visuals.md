# 🎨 Fase 4: Explicabilidad Médica (XAI) con Grad-CAM

Para garantizar la transparencia del sistema de triaje, implementamos **Grad-CAM** sobre la última capa convolucional de la DenseNet121 Fine-Tuned. Además, para estabilizar las activaciones, utilizamos un **TTA-GradCAM fusionado** (promediando los mapas de calor de la imagen original, rotada y con zoom).

### 🔍 Hallazgos de la Auditoría Visual
Al analizar las predicciones correctas de Neumonía, el mapa de calor revela un comportamiento crítico del modelo:

1.  **Activación de Patología:** El modelo identifica correctamente las zonas de infiltración y consolidación de líquido dentro de la cavidad pulmonar.
2.  **Aprendizaje de Atajos (Shortcut Learning):** Se observan fuertes activaciones espurias en la zona de los hombros, clavículas y brazos. 

### 🧠 Análisis del Sesgo (Efecto Clever Hans)
Estas activaciones periféricas indican que el modelo está utilizando variables de confusión para predecir. En la práctica clínica, los pacientes sanos suelen ser escaneados de pie (vista PA), mientras que los pacientes críticos son escaneados en decúbito supino (tumbados, vista AP). La red neuronal ha aprendido a correlacionar la geometría de los brazos y hombros asociada a la postura de la cama con la presencia de neumonía.

### 🚀 Trabajo Futuro (Next Steps)
Para un despliegue de nivel FDA/CE, el *pipeline* deberá incluir un **modelo de segmentación semántica previo (ej. U-Net)** que recorte y aísle exclusivamente la caja torácica anatómica antes de pasar la imagen al modelo de clasificación. Esto forzará a la DenseNet a basar su decisión al 100% en el parénquima pulmonar, eliminando el sesgo postural.