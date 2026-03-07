# 🫁 Registro de Experimentación del Modelo: Detección de Neumonía

## 📌 Contexto Clínico y Objetivo de Optimización
El dataset de Kaggle (Paul Mooney) presenta un fuerte desbalance natural de clases (aproximadamente 3:1 a favor de imágenes con neumonía frente a normales). 

En un entorno médico, la asimetría del error es crítica:
* **Falso Positivo (FP):** Diagnosticar neumonía a un paciente sano. (Coste: pruebas adicionales, ansiedad temporal).
* **Falso Negativo (FN):** Mandar a casa a un paciente con neumonía diagnosticándolo como sano. (Coste: riesgo grave para la salud).

**Objetivo de la experimentación:** Encontrar el equilibrio óptimo (*trade-off*) ajustando la función de pérdida (`CrossEntropyLoss`) para priorizar el **Recall** (minimizar Falsos Negativos), manteniendo una precisión general aceptable.

---

## 🧪 Experimento A: Baseline (Pesos Neutros)

### ⚙️ Configuración
* **Arquitectura:** ResNet18 (Transfer Learning, base congelada).
* **Optimizador:** Adam (lr=0.001) entrenando solo la capa `fc`.
* **Función de Pérdida:** `CrossEntropyLoss` con pesos `[1.0, 1.0]`.
* **Hipótesis:** Al no aplicar penalizaciones manuales, el modelo dependerá únicamente de la distribución natural del dataset. Esperamos un Accuracy general alto, pero un número inaceptable de Falsos Negativos debido a la falta de sesgo clínico.

### 📊 Resultados (Epoch 5)
* **Loss (Train / Val):** 0.1518 / 0.1011
* **Accuracy General:** 95.99%

| Métrica Clínica | Valor | Implicación en el Mundo Real |
| :--- | :--- | :--- |
| **Falsos Negativos (FN)** | 19 | ⚠️ 19 enfermos graves sin tratamiento. |
| **Falsos Positivos (FP)** | 23 | 💸 23 sanos sometidos a más pruebas. |
| **Aciertos Neumonía (TP)** | 758 | Diagnósticos correctos de enfermedad. |
| **Aciertos Normales (TN)** | 247 | Pacientes sanos dados de alta correctamente. |

### 🧠 Análisis (Conclusión del Exp. A)
El experimento sirve como un *baseline* excepcional. Alcanzar casi un 96% de precisión en solo 5 épocas demuestra la potencia del Transfer Learning. El modelo no sufre de *overfitting* (la pérdida de validación es menor que la de entrenamiento). 

Sin embargo, desde el punto de vista del producto, **19 Falsos Negativos es una cifra de riesgo**. El modelo es demasiado "equilibrado". Necesitamos sesgarlo para que sea más conservador a la hora de predecir que un paciente está "Sano".

---

## 🧪 Experimento B: Balanceo de Clases Matemático (Pesos a la Minoría)

### ⚙️ Configuración
* **Arquitectura:** ResNet18 (Transfer Learning, base congelada, nueva capa `fc` reseteada).
* **Optimizador:** Adam (lr=0.001) entrenando solo la capa `fc`.
* **Función de Pérdida:** `CrossEntropyLoss` con pesos `[4.0, 1.0]` (Normal: 4.0, Neumonía: 1.0).
* **Hipótesis:** El dataset tiene un desbalance natural masivo (aprox. 3100 imágenes de Neumonía frente a 1000 Normales). En este experimento, aplicamos un peso de `4.0` a la clase minoritaria ("Normal") para forzar a la red a prestarle la misma atención matemática. Esperamos que el modelo mejore en la detección de pulmones sanos (reduciendo Falsos Positivos), pero debemos vigilar el impacto en los Falsos Negativos clínicos.

### 📊 Resultados (Epoch 5)
* **Loss (Train / Val):** 0.1544 / 0.1256
* **Accuracy General:** 95.32%

| Métrica Clínica | Valor | Implicación en el Mundo Real | Comparativa vs Exp A |
| :--- | :--- | :--- | :--- |
| **Falsos Negativos (FN)** | 34 | ⚠️ 34 enfermos graves sin tratamiento. | ❌ Empeora (+15) |
| **Falsos Positivos (FP)** | 15 | 💸 15 sanos sometidos a más pruebas. | ✅ Mejora (-8) |
| **Aciertos Neumonía (TP)** | 743 | Diagnósticos correctos de enfermedad. | ⬇️ Disminuye |
| **Aciertos Normales (TN)** | 255 | Pacientes sanos dados de alta correctamente. | ⬆️ Aumenta |

### 🧠 Análisis (Conclusión del Exp. B)
**Éxito matemático, fracaso clínico.** El experimento ha funcionado exactamente como dicta la teoría matemática: al penalizar severamente los errores en la clase "Normal", el modelo se ha vuelto un experto en detectar pulmones sanos. Los Falsos Positivos han bajado (de 23 a 15) y los aciertos en la clase Normal han subido. 

Sin embargo, en el contexto médico, el resultado es inaceptable. Como el modelo ahora le tiene "pánico" a equivocarse con un paciente sano, se ha vuelto extremadamente conservador a la hora de diagnosticar Neumonía. Ante la más mínima duda, prefiere decir "Normal" para evitar el castigo x4. Esto ha disparado los Falsos Negativos de 19 a 34 (hemos mandado a casa a 15 enfermos más que en el *baseline*).

**Próximos pasos:** Este experimento demuestra que "balancear un dataset" no siempre es la solución correcta si la métrica de negocio exige asimetría. Queda claro que nuestro objetivo no es el balance matemático, sino el sesgo clínico hacia la sensibilidad. Para el **Experimento C**, jugaremos a la inversa: daremos un "ligero empujón" clínico a la Neumonía (ej. `[1.0, 1.5]`) para forzar al modelo a detectar más enfermos, buscando bajar los Falsos Negativos sin que los gradientes se rompan como ocurrió en nuestros planteamientos iniciales.

---

## 🧪 Experimento C: Hyperparameter Tuning de Pesos Clínicos (Grid Search)
### ⚙️ Configuración y Objetivo
* **Arquitectura:** ResNet18.
* **Técnica:** Variación iterativa de la función de pérdida (`CrossEntropyLoss`).
* **Descripción:** Tras observar que el balance matemático para compensar la asimetria del dataset (Exp B) no mejora los resultados de preción general y aumenta el numero de *Falsos Negativos*, exploraremos pequeñas penalizaciones a la clase Neumonía para forzar un sesgo clínico hacia el Recall (minimizar Falsos Negativos), con el objetivo de disminuir los falsos Negativos a un número menor que en el *baseline*. 

### 📌 Sub-experimento C1: Empuje Clínico Suave (Pesos 1.0 / 1.5)

### ⚙️ Configuración
* **Arquitectura:** ResNet18 (Transfer Learning, base congelada, nueva capa `fc` reseteada).
* **Optimizador:** Adam (lr=0.001) entrenando solo la capa `fc`.
* **Función de Pérdida:** `CrossEntropyLoss` con pesos `[1.0, 1.5]` (Normal: 1.0, Neumonía: 1.5).
* **Hipótesis:** Sabiendo que penalizar masivamente a la clase mayoritaria rompe los gradientes (visto en experimentos previos), probamos un enfoque mucho más suave. El objetivo es dar un ligero sesgo clínico hacia la Neumonía sin desestabilizar la red, buscando bajar los Falsos Negativos del *baseline* (19) a un número menor.

### 📊 Resultados (Epoch 5)
* **Loss (Train / Val):** 0.1299 / 0.1194
* **Accuracy General:** 95.61%

| Métrica Clínica | Valor | Implicación en el Mundo Real | Comparativa vs Exp A (Baseline) |
| :--- | :--- | :--- | :--- |
| **Falsos Negativos (FN)** | 25 | ⚠️ 25 enfermos graves sin tratamiento. | ❌ Empeora (+6) |
| **Falsos Positivos (FP)** | 21 | 💸 21 sanos sometidos a más pruebas. | ✅ Mejora (-2) |
| **Aciertos Neumonía (TP)** | 752 | Diagnósticos correctos de enfermedad. | ⬇️ Disminuye |
| **Aciertos Normales (TN)** | 249 | Pacientes sanos dados de alta correctamente. | ⬆️ Aumenta |

### 🧠 Análisis (Conclusión del Exp. C1)
Aumentar la penalización sobre los datos de Nuemonia no ha ayudado a disminuir Falsos Negativos. Incluso un "empujoncito" suave (1.5) a la clase que ya es mayoritaria por naturaleza resulta contraproducente para nuestro objetivo clínico. 

En lugar de bajar los Falsos Negativos, han subido de 19 a 25 respecto a nuestro modelo de pesos neutros. Esto significa que manipular la `CrossEntropyLoss` a favor de la clase ya dominante desorienta al optimizador. Curiosamente, la red se ha vuelto *mejor* detectando pacientes sanos (los Falsos Positivos han bajado de 23 a 21), pero hemos perdido sensibilidad clínica.

**Conclusión:** Tocar los pesos de la función de pérdida no es la herramienta adecuada para este dataset específico. Queda demostrado que el entrenamiento más estable para extraer los patrones correctos es el de los pesos neutros (`[1.0, 1.0]`). Aun así aprovechando que el entrenamiento no es costoso a nivel computacional, vamos a intentar comprobar el efecto de una penalización ligeramente menor para la Neumonia.

### 📌 Sub-experimento C2: Empuje Clínico Ligero (Pesos 1.0 / 1.2)

### ⚙️ Configuración
* **Arquitectura:** ResNet18 (Transfer Learning, base congelada, nueva capa `fc` reseteada).
* **Optimizador:** Adam (lr=0.001) entrenando solo la capa `fc`.
* **Función de Pérdida:** `CrossEntropyLoss` con pesos `[1.0, 1.2]` (Normal: 1.0, Neumonía: 1.2).
* **Hipótesis:** Aprovechando el bajo coste computacional, decidimos trazar la curva de degradación comprobando un peso intermedio muy sutil. Si `1.5` rompía la estabilidad y `1.0` era estable, probamos `1.2` para ver si este "micro-ajuste" logra reducir los Falsos Negativos sin desorientar al optimizador.

### 📊 Resultados (Epoch 5)
* **Loss (Train / Val):** 0.1157 / 0.1070
* **Accuracy General:** 96.47% *(Mayor precisión general hasta este momento)*

| Métrica Clínica | Valor | Implicación en el Mundo Real | Comparativa vs Exp A (Baseline) |
| :--- | :--- | :--- | :--- |
| **Falsos Negativos (FN)** | 20 | ⚠️ 20 enfermos graves sin tratamiento. | ↔️ Prácticamente igual (+1) |
| **Falsos Positivos (FP)** | 17 | 💸 17 sanos sometidos a más pruebas. | ✅ Mejora (-6) |
| **Aciertos Neumonía (TP)** | 757 | Diagnósticos correctos de enfermedad. | ↔️ Prácticamente igual (-1) |
| **Aciertos Normales (TN)** | 253 | Pacientes sanos dados de alta correctamente. | ⬆️ Aumenta |

### 🧠 Análisis (Conclusión del Exp. C2)
La intuición de probar este peso ligero ha sido un éxito técnico: hemos conseguido el modelo más preciso hasta la fecha (96.47% de Accuracy) y una caída fantástica en la pérdida de validación (0.1070). El modelo ha encontrado un equilibrio excelente, reduciendo los Falsos Positivos (de 23 a 17).

Sin embargo, desde la perspectiva estricta de nuestro objetivo (minimizar los Falsos Negativos), no hemos conseguido mejoras. 20 Falsos Negativos es esencialmente un empate técnico con los 19 de nuestro *baseline*. Todo apunta que la función de pérdida no da más de sí para forzar el *Recall* en este dataset sin destruir la precisión general.

**Conclusión definitiva sobre los pesos:** El rango óptimo de entrenamiento para este dataset desbalanceado está entre `[1.0, 1.0]` y `[1.0, 1.2]`. 

**Próximos pasos:** Damos por concluida la fase de optimización durante el entrenamiento (*Training-time tuning*). Para el **Experimento D**, cargaremos este modelo C2 en memoria y aplicaremos **Threshold Tuning** (Ajuste de Umbral). Analizaremos las probabilidades brutas (Softmax) en la fase de inferencia y bajaremos el umbral de decisión de Neumonía del 50% al 30% o 20% para forzar artificialmente la caída de esos 20 Falsos Negativos, intentando no afectar demasiado a la precisión general.

---

## 🧪 Experimento D: Threshold Tuning (Calibración del Umbral de Decisión)

### ⚙️ Configuración
* **Modelo Base:** El modelo más preciso obtenido (Exp C2 - Pesos 1.0 / 1.2).
* **Técnica:** Modificación probabilística post-entrenamiento en fase de inferencia.
* **Nuevo Umbral:** `0.30` (30%). En lugar de exigir un 50% de seguridad matemática, forzamos al modelo a diagnosticar Neumonía si detecta al menos un 30% de probabilidad en la imagen.
* **Hipótesis:** Ajustar el límite de decisión a posteriori es mucho más estable que forzar los gradientes de pérdida durante el entrenamiento. Esperamos una caída drástica de Falsos Negativos asumiendo un repunte controlado de Falsos Positivos.

### 📊 Resultados (Con Umbral al 30%)
* **Accuracy General:** 95.61%

| Métrica Clínica | Valor | Implicación en el Mundo Real | Comparativa vs Mejor Modelo (Exp C2) |
| :--- | :--- | :--- | :--- |
| **Falsos Negativos (FN)** | **13** | ⚠️ 13 enfermos graves sin tratamiento. | 🌟 **Mejora masiva (-7)** |
| **Falsos Positivos (FP)** | 33 | 💸 33 sanos sometidos a más pruebas. | 📉 Empeora (+16) |
| **Aciertos Neumonía (TP)** | 764 | Diagnósticos correctos de enfermedad. | ⬆️ Aumenta (+7) |
| **Aciertos Normales (TN)** | 237 | Pacientes sanos dados de alta. | ⬇️ Disminuye (-16) |

### 🧠 Análisis (Conclusión del Exp. D)
Este experimento demuestra que separar la optimización matemática (entrenamiento) de la optimización clinica (evaluación) es buena estrategia para un dataset desbalanceado como este. 

Mantuvimos el "cerebro" estable y preciso del Experimento C2, pero cambiamos sus instrucciones de diagnóstico. El resultado es el modelo más seguro y viable para un entorno médico real que hemos conseguido hasta ahora, sacrificando apenas un 0.8% de *Accuracy* general a cambio de reducir los errores fatales (FN) casi a la mitad.

### 📌 Sub-experimento D2: Umbral Clínico Agresivo (20%)

### ⚙️ Configuración
* **Modelo Base:** Exp C2 (Pesos 1.0 / 1.2).
* **Nuevo Umbral:** `0.20` (20%). Si el modelo detecta un 20% de características compatibles con Neumonía, lanza la alerta.

### 📊 Resultados (Con Umbral al 20%)
* **Accuracy General:** 94.84%

| Métrica Clínica | Valor | Implicación en el Mundo Real | Comparativa vs Mejor Modelo (Exp C2) |
| :--- | :--- | :--- | :--- |
| **Falsos Negativos (FN)** | **7** | ⚠️ 7 enfermos graves sin tratamiento. | 🏆 **Mejora récord (-13)** |
| **Falsos Positivos (FP)** | 47 | 💸 47 sanos sometidos a más pruebas. | 📉 Empeora (+30) |
| **Aciertos Neumonía (TP)** | 770 | Diagnósticos correctos. (Sensibilidad: 99.1%) | ⬆️ Aumenta (+13) |
| **Aciertos Normales (TN)** | 223 | Pacientes sanos dados de alta. | ⬇️ Disminuye (-30) |

### 🧠 Análisis (Conclusión del Exp. D2)
Bajar el umbral de decisión al 20% ha demostrado ser una estrategia muy efectiva. Hemos logrado una **Sensibilidad del 99.1%**, reduciendo los Falsos Negativos a una cifra de un solo dígito (7). 

Aunque el *Accuracy* general bajó al 94.84% y los Falsos Positivos aumentaron a 47, este es el *trade-off* (intercambio) exacto que requiere un sistema de triaje médico. El coste de revisar manualmente a 47 pacientes sanos es infinitamente menor que el coste vital de omitir a 13 pacientes enfermos adicionales (la diferencia con nuestro modelo anterior).

**Veredicto:** El pipeline final podría utilizará los pesos de entrenamiento del Experimento C2 (`[1.0, 1.2]`) combinados con un filtro de inferencia basado en un umbral probabilístico del `0.20`.

---

## 🧪 Experimento E: Test-Time Augmentation (TTA) y Consenso (Votación Mayoritaria)

### ⚙️ Configuración
* **Modelo Base:** El mejor modelo en pesos (Exp C2 - Pesos 1.0 / 1.2).
* **Técnica:** TTA (Test-Time Augmentation) con "Hard Voting".
* **Descripción:** En fase de inferencia, no pasamos la imagen una sola vez. Generamos 3 variantes en tiempo real por cada paciente:
  1. Imagen Original (Resize + Normalize).
  2. Imagen Rotada (10 grados).
  3. Imagen con Zoom (CenterCrop).
* **Regla de decisión:** El modelo emite 3 votos independientes. Si la suma de votos positivos (Neumonía) es $\ge 2$, el diagnóstico final es Neumonía.

### 📊 Resultados (Epoch 5 + Inferencia TTA)
* **Accuracy General:** 97.23% ⭐ *(Récord absoluto del proyecto)*

| Métrica Clínica | Valor | Implicación en el Mundo Real | Comparativa vs Mejor Modelo (Exp C2) |
| :--- | :--- | :--- | :--- |
| **Falsos Negativos (FN)** | **6** | ⚠️ 6 enfermos graves sin tratamiento. | 🏆 **Mejora récord (-14)** |
| **Falsos Positivos (FP)** | 23 | 💸 23 sanos sometidos a más pruebas. | 📉 Empeora ligeramente (+6) pero iguala al baseline. |
| **Aciertos Neumonía (TP)** | 771 | Diagnósticos correctos. (Sensibilidad: **99.2%**) | ⬆️ Aumenta (+14) |
| **Aciertos Normales (TN)** | 247 | Pacientes sanos dados de alta. | ⬇️ Disminuye (-6) |

### 🧠 Análisis (COnclusión Inferencia TTA)
Este experimento demuestra que aumentar el esfuerzo en la fase de inferencia puede superar a las optimizaciones del entrenamiento. 

Al forzar un consenso de 3 vías (TTA), el modelo ha demostrado una robustez espectacular frente a posibles sesgos de posición o encuadre en las radiografías. Hemos logrado la mejor métrica clínica (Sensibilidad del 99.2%, con solo 6 Falsos Negativos) manteniendo los Falsos Positivos a raya y logrando el pico máximo de *Accuracy* general del proyecto (97.23%).

**Veredicto** El sistema de producción (*Backend*) implementará el modelo C2 (`[1.0, 1.2]`) y utilizará esta 3-transformation pipeline y votación mayoritaria para cada nueva radiografía.



---

## 🧪 Experimento F: Focal Loss (Penalización Dinámica de Errores)
### ⚙️ Configuración y Objetivo
* **Técnica:** Cambio de Función de Pérdida a *Focal Loss*.
* **Descripción:** Dejaremos de usar `CrossEntropyLoss` y los pesos estáticos. Implementaremos *Focal Loss*, una función diseñada específicamente para datasets desbalanceados. Esta función reduce dinámicamente el peso de las imágenes "fáciles" y concentra toda la atención matemática de la red en los ejemplos "difíciles" (los Falsos Negativos históricos), obligando al modelo a aprender patrones más complejos.

---

## 🧪 Experimento G: Arquitectura Especializada (DenseNet121)
### ⚙️ Configuración y Objetivo
* **Técnica:** Cambio de Arquitectura Base (Transfer Learning).
* **Descripción:** ResNet18 es rápido y eficiente, pero DenseNet121 es el *gold standard* en la literatura médica (utilizado en el modelo CheXNet de la Universidad de Stanford). Sus conexiones densas entre capas preservan mejor la información de alta frecuencia (texturas sutiles), lo cual es crítico para detectar infiltraciones pulmonares borrosas que ResNet podría pasar por alto.