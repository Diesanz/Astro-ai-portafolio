# Informe de Preparación y Análisis de Datos: Caídas de Meteoritos

## 1. Introducción

Este informe documenta el proceso de exploración, limpieza y preparación de un conjunto de datos sobre caídas de meteoritos en la Tierra. El objetivo es dejar los datos listos para su uso en modelos de clasificación y análisis posteriores.

---

## 2. Importación de Librerías

Se utilizaron las siguientes librerías:
- **pandas** para manipulación de datos.
- **matplotlib** para visualización.
- **scikit-learn** para preprocesamiento y transformación de datos.

---

## 3. Lectura y Exploración Inicial

El dataset original se cargó desde `./Datasets/meteorite-landings.csv`. Se revisaron las dimensiones y una muestra de los datos, así como información general y estadísticas descriptivas para entender la estructura y calidad de los datos.

---

## 4. Visualización de Datos

Se realizaron dos visualizaciones principales:
- **Distribución de meteoritos por año:** Se graficó un histograma para observar la frecuencia de caídas a lo largo del tiempo.
- **Top 5 tipos de meteoritos más comunes:** Se identificaron y graficaron las cinco clases de meteoritos más frecuentes en el dataset.

---

## 5. Limpieza de Datos

Se aplicaron los siguientes filtros y transformaciones:
- **Años válidos:** Solo se consideraron registros entre 860 d.C. y 2016 d.C.
- **Coordenadas válidas:** Se eliminaron registros con coordenadas fuera de rango o nulas.
- **Eliminación de nulos:** Se descartaron filas con valores faltantes en las columnas clave (`year`, `reclat`, `reclong`, `mass`, `recclass`).
- **Reindexado:** Se reiniciaron los índices tras la limpieza.

---

## 6. Preparación para Modelado

- **Selección de características:** Se eligieron columnas relevantes para el modelo (`nametype`, `fall`, `mass`, `reclat`, `reclong`, `year`).
- **Transformación de datos:**
    - Variables categóricas: Codificadas con OneHotEncoder.
    - Variables numéricas: Escaladas con MinMaxScaler.
    - Coordenadas: Normalizadas.
- **Codificación de etiquetas:** La columna objetivo (`recclass`) se codificó numéricamente.

---

## 7. Resultados del Preprocesamiento

- Se muestran las dimensiones finales del dataset limpio.
- Se verifica la cantidad de clases únicas y ejemplos de etiquetas codificadas.

---

## 8. Guardado del Dataset Limpio

El dataset procesado se guardó como `./Datasets/meteorite-landings-limpio.csv` para su uso en análisis y modelos posteriores.

---

## 9. Conclusión

El proceso permitió obtener un conjunto de datos limpio, estructurado y listo para tareas de clasificación y análisis de caídas de meteoritos, facilitando futuros estudios y aplicaciones de inteligencia artificial en el área.
