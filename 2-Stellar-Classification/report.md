# Clasificación Estelar con Redes Neuronales

## Introducción

En este proyecto se aborda la clasificación automática de objetos astronómicos (estrellas, galaxias y cuásares) utilizando un modelo de red neuronal profunda implementado en PyTorch. El objetivo es predecir la clase de cada objeto a partir de sus características fotométricas y el redshift.

---

## Preparación y Limpieza de Datos

- Se utilizó un dataset con las columnas `u`, `g`, `r`, `i`, `z`, `redshift` y `class`.
- Se eliminaron filas con valores nulos y se ajustaron los índices.
- Las características numéricas fueron estandarizadas y las etiquetas codificadas.

---

## División de Datos

- **Entrenamiento:** 60%
- **Validación:** 20%
- **Test:** 20%

---

## Arquitectura del Modelo

- Red neuronal con 3 capas ocultas (256, 128, 128 neuronas).
- Funciones de activación ReLU, normalización por lotes y dropout para evitar overfitting.
- Optimización con Adam y función de pérdida CrossEntropy.

---

## Entrenamiento

- El modelo se entrenó durante 100 épocas.
- Se monitorizó la pérdida y precisión tanto en entrenamiento como en validación.

### Gráficas de Entrenamiento

- **Pérdida:** Disminuye rápidamente y se estabiliza, sin señales de sobreajuste.
- **Precisión:** Alcanza valores altos y similares en entrenamiento y validación.

---

## Evaluación en el Conjunto de Test

- **Precisión global en test:** ~0.98

### Matriz de Confusión

La matriz de confusión muestra que la mayoría de las predicciones están en la diagonal principal, indicando una alta tasa de aciertos para cada clase:

|           | Predicho: GALAXY | Predicho: QSO | Predicho: STAR |
|-----------|------------------|---------------|----------------|
| **Real: GALAXY** | 11604            |  153          |  103           |
| **Real: QSO**    |  120             | 3485          |   15           |
| **Real: STAR**   |   10             |   12          | 4340           |

- La clase STAR tiene una tasa de acierto casi perfecta.
- Los errores más frecuentes ocurren entre GALAXY y QSO, lo cual es común por la similitud espectral.

---

### Reporte de Clasificación

- **Precisión y recall** altos para todas las clases.
- El modelo generaliza bien y no memoriza los datos de entrenamiento.

---

## Conclusiones

- El modelo de red neuronal logra una excelente clasificación de objetos astronómicos.
- La arquitectura y el preprocesamiento permiten distinguir correctamente entre galaxias, cuásares y estrellas.
- El sistema es robusto y puede ser utilizado como base para tareas de clasificación astronómica más avanzadas.

---

## Guardado del Modelo

- El modelo entrenado, el scaler y el label encoder fueron guardados para su uso futuro.
