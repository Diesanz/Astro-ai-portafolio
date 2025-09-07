# Proyecto 4 — Chatbot Astronómico con GPT-2 y RAG

## Objetivo
Desarrollar un chatbot capaz de responder preguntas sobre astronomía utilizando:
- Un modelo GPT-2 afinado (`gpt2-astrotuned`).
- Un sistema RAG (Retrieval-Augmented Generation) que permite que el modelo consulte un conjunto de documentos externos para mejorar la precisión de las respuestas.


---

## Estructura de Archivos
````
├─ 4-Astronomic-Chatbot/
│ ├─ crear_dataset.py # Script para crear y limpiar dataset de textos
│ ├─ rag.py # Clase AstroRag y funciones RAG
│ ├─ gradio_interface.py # Interfaz Gradio para comparación de modelos
│ ├─ gpt2-astrotuned/ # Carpeta con modelo GPT-2 entrenado
│ └─ results/
│ └─ astronomia_corpus.json # Corpus de conocimiento externo
````


---

## Flujo de Trabajo
```
    A[Usuario hace pregunta] --> B{Selecciona modelo}
    B --> |GPT-2| C[GPT-2 afinado genera respuesta]
    B --> |GPT-2 + RAG| D[RAG busca contexto en corpus]
    D --> E[Construye prompt: "Contexto + Pregunta"]
    E --> F[GPT-2 genera respuesta usando contexto]
    C --> G[Mostrar respuesta al usuario]
    F --> G
```
### 1. Entrenamiento del modelo GPT-2
- Crear dataset de preguntas/respuestas o textos astronómicos.
- Limpiar textos (normalizar, quitar URLs, símbolos extraños).
- Afinar GPT-2 usando `Trainer` de Hugging Face.
- Guardar el modelo entrenado en `./gpt2-astrotuned`.

```
    A[Corpus de textos astronómicos] --> B[Limpieza y normalización]
    B --> C[Tokenización con truncation=True]
    C --> D[Entrenamiento GPT-2 con Trainer]
    D --> E[Modelo afinado guardado en ./gpt2-astrotuned]
```

### 2. Implementación de RAG
- Clase `AstroRag`:
  - Limpia y prepara el corpus (`astronomia_corpus.json`).
  - Crea embeddings con `SentenceTransformer`.
  - Indexa los embeddings usando FAISS.
  - Recupera el documento más relevante para una pregunta y genera respuesta con GPT-2 usando ese contexto.
- Permite generar respuestas más informadas usando documentos externos.

```
    A[Corpus: astronomia_corpus.json] --> B[Limpieza de texto (regex, minúsculas)]
    B --> C[Embeddings con SentenceTransformer]
    C --> D[Indexación con FAISS]
    E[Pregunta usuario] --> F[Embedding de la pregunta]
    F --> D
    D --> G[Recuperar documento más relevante]
    G --> H[Construir prompt: "Contexto + Pregunta"]
    H --> I[Generar respuesta con GPT-2]
    I --> J[Mostrar respuesta al usuario]
```

### 3. Comparación GPT-2 vs GPT-2 + RAG
- Se puede cargar el GPT-2 simple entrenado y el modelo RAG.
- Comparar la precisión y coherencia de las respuestas.
- Observación: GPT-2 a veces da respuestas más coherentes, RAG añade contexto pero puede generar frases largas o truncadas.

### 4. Interfaz Gradio
- Permite hacer preguntas al chatbot.
- Comparar ambos modelos:
  - **GPT-2**
  - **GPT-2 + RAG**
- Parámetros ajustables:
  - `top_k`: número de tokens candidatos considerados.
  - `max_context_tokens`: tokens máximos tomados del contexto (solo RAG).
  - `max_new_tokens`: tokens máximos a generar en la respuesta.
  - `temperature`: creatividad de la respuesta (0 determinista, >0 más creativo).

---

## Recomendaciones de Configuración 
- `top_k`: 50
- `max_context_tokens`: 200 (solo RAG)
- `max_new_tokens`: 50 (incluso ás para GPT2 ya que añade más información y es más coherente)
- `temperature`: 0.7
- GPU activada si está disponible para acelerar generación.
- Para evitar truncamiento: usar `truncation=True` al tokenizar el corpus y `max_new_tokens` para controlar longitud de generación.

---

## Observaciones Finales
- RAG permite que GPT-2 consulte documentos, pero puede generar frases largas, repetitivas o truncadas.
- GPT-2 afinado solo puede ser más coherente pero sin acceso a contexto externo.
- El proyecto queda modular:
  - `crear_dataset.py`: preparación de corpus.
  - `rag.py`: clase AstroRag.
  - `gradio_interface.py`: interfaz para pruebas y ajustes.
- Perfecto para retomar en el futuro o añadir más documentos y mejorar respuestas.

---

## Cómo Lanzar
1. Entrenar GPT-2 y guardar en `./gpt2-astrotuned`.
2. Ejecutar `rag.py` para probar RAG.
3. Ejecutar `gradio_interface.py` para probar la interfaz y comparar modelos:
   ```bash
   conda activate astro_env
   python gradio_interface.py

---

## Conclusiones y Mejoras

### Conclusiones
- El proyecto demuestra que un **GPT-2 afinado** puede generar respuestas coherentes sobre astronomía, aunque sin acceso a información externa.
- El uso de **RAG** permite que el modelo consulte un corpus de documentos, proporcionando contexto adicional y mejorando potencialmente la precisión en temas específicos.
- Sin embargo, se observa que RAG puede generar respuestas más largas, repetitivas o con frases truncadas, mientras que GPT-2 solo puede ser más coherente pero limitado a lo aprendido durante el entrenamiento.
- La comparación entre ambos modelos resalta la **importancia del contexto externo** y la necesidad de balancear la creatividad (`temperature`) y el número de tokens generados (`max_new_tokens`) para obtener respuestas útiles y precisas.

### Posibles Mejoras
1. **Optimización del corpus**: agregar más documentos, resúmenes o artículos recientes de astronomía para que RAG tenga más información relevante.
2. **Filtrado y control de longitud**: implementar truncamiento más inteligente o resúmenes automáticos de documentos para evitar respuestas muy largas o repetitivas.
3. **Afinación de parámetros dinámicos**: permitir que el usuario ajuste `top_k`, `max_context_tokens`, `max_new_tokens` y `temperature` para adaptar la generación a distintos tipos de preguntas.
4. **Normalización y capitalización**: corregir problemas como la primera letra en minúscula o frases incongruentes para mejorar legibilidad.
5. **Mejoras en embeddings**: explorar modelos de embeddings más grandes o específicos de dominio para mejorar la recuperación de documentos.
6. **Evaluación cuantitativa**: medir precisión y coherencia usando métricas objetivas sobre un conjunto de preguntas de prueba para evaluar la efectividad de GPT-2 vs RAG.

> Este proyecto queda modular y listo para expansión, permitiendo incorporar más conocimiento astronómico, mejoras en RAG y ajuste fino de parámetros para un chatbot más preciso y natural.
