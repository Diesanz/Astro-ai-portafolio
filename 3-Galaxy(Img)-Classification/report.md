# Proyecto 3 — Clasificación de Galaxias (Galaxy Zoo)

Este informe documenta el proyecto de clasificación de galaxias mediante redes neuronales convolucionales (CNN), empleando el dataset Galaxy Zoo. Se describe el objetivo, datos, metodología, configuración de entrenamiento, métricas, resultados actuales, proceso de inferencia (submission) y próximos pasos para mejorar rendimiento y reproducibilidad.

---

## 1. Objetivo
Entrenar un modelo de Deep Learning para predecir las probabilidades de morfologías galácticas (37 columnas de salida) a partir de imágenes, con evaluación mediante RMSE (global y por etiqueta), y generar un archivo de predicción para el subconjunto de test en el formato exigido por la competición de Kaggle (Galaxy Zoo: The Galaxy Challenge).

---

## 2. Datos
- Fuente: Galaxy Zoo (Kaggle) — Galaxy Zoo: The Galaxy Challenge
- Estructura local empleada:
  - Datasets/
    - images_training_rev1/ → imágenes de entrenamiento
    - images_test_rev1/ → imágenes de test
    - training_solutions_rev1.csv → etiquetas de entrenamiento (37 columnas por imagen)
    - all_zeros_benchmark.csv → plantilla de submission que define el orden requerido
- Consideraciones:
  - La evaluación se basa en la comparación de probabilidades predichas frente a las etiquetas reales (regresión multisalida, no clasificación discreta).
  - Es crítico respetar el orden de filas del benchmark (all_zeros_benchmark.csv) al generar el submission para evitar desalineaciones.

---

## 3. Estructura del proyecto
```
3-Galaxy(Img)-Classification/
├─ clasificacion_galaxias.ipynb        # Notebook principal (entrenamiento/inferencia)
├─ extraccion_datos.ipynb              # Notebook auxiliar (si aplica)
├─ Datasets/
│  ├─ images_training_rev1/
│  ├─ images_test_rev1/
│  ├─ training_solutions_rev1.csv
│  └─ all_zeros_benchmark.csv
└─ results/
   ├─ galaxycnn_weights.pth            # Pesos del modelo (último/best)
   └─ submission.csv                   # Submission generado
```

---

## 4. Metodología
- Preprocesamiento de imágenes:
  - Carga mediante PyTorch y torchvision.transforms.
  - Redimensionado y normalización
  - Opcional (recomendado): aumentos de datos (flips, rotaciones suaves, RandomResizedCrop).

- Formulación del problema:
  - Regresión multietiqueta con 37 salidas (probabilidades por condición/etiqueta).
  - Función de pérdida: MSELoss (o SmoothL1Loss); Métrica de evaluación: RMSE (global y por columna).

- Modelo (GalaxyCNN):
  - Arquitectura CNN propia definida en el notebook.
  - Capas:
    - conv1: Conv2d(3→32, k=3, pad=1) + ReLU; pool1: MaxPool2d(2×2)
    - conv2: Conv2d(32→64, k=3, pad=1) + ReLU; pool2: MaxPool2d(2×2)
    - conv3: Conv2d(64→128, k=3, pad=1) + ReLU; pool3: MaxPool2d(2×2)
    - conv4: Conv2d(128→256, k=3, pad=1) + ReLU; pool4: AvgPool2d(2×2)
    - Flatten
    - fc1: Linear(256·Hf·Wf → 512) + ReLU, donde Hf y Wf son las dimensiones espaciales tras los pools
    - out: Linear(512 → 37) con activación lineal (las probabilidades se optimizan con MSE sobre targets en [0,1])
  - Explicación rápida de las capas:
    - Convolución + ReLU: extrae patrones locales (bordes, texturas) y añade no linealidad para aprender representaciones complejas.
    - Pooling: reduce resolución y aporta invariancia traslacional; MaxPool resalta activaciones máximas, AvgPool promedia.
    - Flatten + Fully Connected: combinan las características espaciales en una representación global para predecir las 37 salidas.
  - Recomendación: probar transfer learning (ResNet/EfficientNet/ConvNeXt) para baseline fuerte.

- Entrenamiento:
  - Optimizador (p. ej. Adam/AdamW), LR inicial, scheduler (p. ej. CosineAnnealing/OneCycle), epochs.
  - Entrenamiento con GPU recomendado y AMP (torch.cuda.amp) para estabilidad/velocidad.

- Validación:
  - Holdout o K-Fold (recomendado: MultilabelStratifiedKFold para balance por etiquetas).
  - Reporte de RMSE global y por etiqueta en validación.

- Post-procesado (constraints):
  - En Galaxy Zoo existen grupos de respuestas que deben sumar 1 por “pregunta”.
  - Aplicar normalización por grupos y enmascarado según reglas del árbol de decisión mejora la coherencia del vector de salida.

---

## 5. Configuración y reproducibilidad
- Versiones y dependencias principales (ver requirements.txt a nivel de repo):
  - Python 3.11+, PyTorch 2.3.1 (CUDA 12.1), torchvision 0.18.1, torchaudio 2.3.1
  - numpy, pandas, scikit-learn, matplotlib, pillow, opencv-python
- Reproducibilidad:
  - Fijar semillas (random, numpy, torch, torch.cuda) y activar cudnn.deterministic cuando aplique.
  - Persistir splits de validación (train_idx/val_idx) para replicar resultados.

---

## 6. Métricas y resultados actuales
- Métrica objetivo: RMSE (global) y RMSE por columna.
- Artefactos actuales:
  - results/galaxycnn_weights.pth → pesos del modelo.
  - results/submission.csv → predicciones para test en el orden del benchmark.
- Resultados obtenidos:
  - Mejor Val Loss (MSE): 0.2351 (época 23/25)
  - Mejor Val CORR (Pearson): 0.7225 (época 25/25)
  - Observaciones:
    - La pérdida de validación desciende de ~0.2553 a ~0.2351 a lo largo del entrenamiento.
    - Se observaron NaN puntuales en la métrica de correlación (ép. 15 y 24).
    - No se registró RMSE explícito en el notebook; actualmente se optimiza MSE y se reporta Pearson.


---

## 7. Generación de submission
- Procedimiento seguido en el notebook:
  1) Cargar el benchmark (all_zeros_benchmark.csv) para fijar el orden de ids.
  2) Preprocesar las imágenes de test con los mismos transforms que train/val.
  3) Inferir con el modelo (modo eval, sin gradientes) y recoger las 37 probabilidades por imagen.
  4) Opcional (recomendado): aplicar TTA (Test-Time Augmentation) y promediar.
  5) Aplicar post-procesado de constraints por grupos (sumas a 1 y máscaras).
  6) Volcar las predicciones a results/submission.csv con el mismo orden que el benchmark.

- Archivo resultante: results/submission.csv

---

## 8. Cómo ejecutar
1) Instalar dependencias (desde la raíz del repositorio):
   ```bash
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
   ```
2) Preparar datos en la carpeta Datasets/ (imágenes y CSVs conforme a la estructura indicada).
3) Abrir y ejecutar el notebook:
   - 3-Galaxy(Img)-Classification/clasificacion_galaxias.ipynb
   - Secciones: carga de datos → definición de modelo → entrenamiento → validación → inferencia/submission
4) Salidas esperadas:
   - Pesos: results/galaxycnn_weights.pth
   - Submission: results/submission.csv

---

## 9. Limitaciones y próximos pasos
- Implementar y reportar RMSE por-columna y global en validación; guardar metrics.json y curvas.
- Reforzar reproducibilidad (semillas, splits persistidos, config.yaml).
- Probar arquitecturas preentrenadas (ResNet/EfficientNet/ConvNeXt) con fine-tuning, AMP y schedulers.
- Añadir augmentations y TTA; evaluar K-Fold y ensembling de folds/checkpoints.
- Integrar post-procesado de constraints de Galaxy Zoo para coherencia de salidas.
- Añadir logging (TensorBoard/Weights & Biases) para trazabilidad de experimentos.

---

## 10. Referencias
- Galaxy Zoo: The Galaxy Challenge (Kaggle)
- Documentación PyTorch y torchvision
- Discusiones de constraints/normalizaciones por grupo en kernels/notebooks de Galaxy Zoo

---

## 11. Autoría
Proyecto desarrollado por Diego dentro del portafolio Astro-AI.
