# Astro-AI Portfolio

Portafolio en el que se desarrollan y documentan diferentes modelos de **redes neuronales** aplicados al **aprendizaje automático en astrofísica**.  
El objetivo es como el *Deep Learning* pueden utilizarse en problemas científicos relacionados con el espacio.

---

## Proyectos incluidos

### 1. Preparación de Datos de Meteoritos
- **Descripción:** Proyecto dedicado a la limpieza, exploración y preparación de datos del catálogo de caídas y hallazgos de meteoritos.  
- **Dataset:** [NASA Meteoritical Bulletin Database](https://www.kaggle.com/datasets/nasa/meteorite-landings).  
- **Objetivo de aprendizaje:**  
  - Recolección y limpieza de datos reales.  
  - Manejo de valores faltantes y outliers.  
  - Filtrado de rangos temporales y geográficos.  
  - Normalización y codificación de variables categóricas.  
- [Preparación de Meteorite Landings](./1-Meteorites-Data-Preparation/)

---

### 2. Clasificación Estelar
- **Descripción:** Clasificación de objetos astronómicos (galaxias, estrellas, cuásares) a partir de datos fotométricos y espectroscópicos.  
- **Dataset:** [Sloan Digital Sky Survey (SDSS17)](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17).  
- **Objetivo de aprendizaje:**  
  - Construcción de un modelo de red neuronal densa (MLP).  
  - Normalización de datos y preprocesamiento con `scikit-learn`.  
  - Entrenamiento y evaluación supervisada en datasets tabulares.  
- [Clasificación Estelar](./2-Stellar-Classification/) 

### 3. Clasificación de Galaxias
- **Descripción:** Entrenamiento de una CNN para clasificar galaxias en distintas morfologías (espiral, elíptica, irregular, etc.).
- **Dataset:** [Galaxy Zoo](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge).
- **Objetivo de aprendizaje:** Procesado de imágenes astronómicas y calsificacion de galaxias.
- [Clasificación Galaxias](./3-Galaxy(Img)-Classification/) 
---

### 3. Detección de Exoplanetas
- **Descripción:** Uso de series temporales de brillo estelar (curvas de luz) para detectar posibles exoplanetas.
- **Dataset:** Datos de las misiones [Kepler](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data) / [TESS](https://exoplanetarchive.ipac.caltech.edu/).
- **Objetivo de aprendizaje:** Redes neuronales recurrentes (RNN, LSTM) y modelos 1D-CNN aplicados a datos temporales.

--- 
### 4. Chatbot Astronómico con LLM y RAG
- **Descripción:** Desarrollo de un chatbot capaz de responder preguntas sobre astronomía y espacio. Se basa en un modelo de lenguaje preentrenado (LLM) combinado con RAG (Retrieval-Augmented Generation), que permite recuperar información de documentos astronómicos reales (Wikipedia, papers de la NASA, datasets abiertos) para dar respuestas más precisas y contextualizadas.
- **Dataset:** Corpus de textos astronómicos (Wikipedia, papers de NASA/arXiv, datasets abiertos de astronomía).
- **Objetivo de aprendizaje:** Uso de LLMs, embeddings semánticos, bases vectoriales y pipelines RAG, junto con el despliegue en una demo interactiva con Gradio o Streamlit.

---

##  Tecnologías utilizadas
- **Lenguaje:** Python 3.11+
- **Librerías principales:**  
  - `torch`, `torchvision` → Deep Learning  
  - `pandas`, `numpy` → Análisis de datos  
  - `matplotlib`, `seaborn` → Visualización  
  - `scikit-learn` → Preprocesamiento y métricas  

### Instalación de paquetes a partir del archivo requirements.txt

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```
---

##  Objetivos del portafolio
1. Aplicar técnicas de aprendizaje automático en problemas científicos reales.  
2. Desarrollar un portafolio demostrable para proyectos de **IA aplicada a la astrofísica**.  
3. Explorar diferentes arquitecturas de redes neuronales (CNN, RNN, MLP).  

---

##  Autor
Proyecto desarrollado por **Diego** como parte de su portafolio personal en *Machine Learning aplicado a la Astrofísica*.  
