
#Objetivo de RAG: La idea es que el modelo no solo "invente" respuestas de memoria, 
#sino que busque información relevante en un conjunto de documentos externos y genere una respuesta usando ese contexto

#Importar librerias necesarias
import warnings
import re
import json
import torch
import numpy as np
import faiss
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from sentence_transformers import SentenceTransformer
from datasets import Dataset

class AstroRag:
    def __init__(self, model_dir="./gpt2-astrotuned", corpus_path="./results/astronomia_corpus.json"):
        warnings.filterwarnings("ignore")

        #Cargar modelo GPT-2 entrenado
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

        #Cargar y limpiar documentos
        self.docs = [self.clean_text(d["text"]) for d in self.get_docs(corpus_path)]
        self.dataset = Dataset.from_dict({"text": self.docs})

        #Crear embeddings para recuperación
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.embedder.encode(self.docs, convert_to_numpy=True)

        #Crear índice FAISS
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

        #Pipeline de generación, asignar GPU si está disponible
        device = 0 if torch.cuda.is_available() else -1
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device
        )

    def get_docs(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def clean_text(self, text):
        #text = text.lower()
        text = re.sub(r"http\S+", "", text)  #quitar URLs
        text = re.sub(r"[^a-zA-Z0-9\s.,;:!?()\-']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def rag_query(self, question, top_k=5, max_context_tokens=200, max_new_tokens=30, temperature=0.6):
        """
        Recupera los documentos más relevantes y genera una respuesta coherente.
        """
        #Obtener embedding de la pregunta
        q_emb = self.embedder.encode([question], convert_to_numpy=True)

        #Recuperar top_k documentos más relevantes
        D, I = self.index.search(q_emb, top_k)
        contexts = [self.dataset[int(idx)]["text"] for idx in I[0]]

        #Limitar tokens del contexto concatenado
        context_text = " ".join(contexts)
        context_tokens = self.tokenizer.encode(context_text, truncation=True, max_length=max_context_tokens)
        context_text = self.tokenizer.decode(context_tokens)

        #Construir prompt más dirigido
        prompt = (
            "Usa SOLO la información del contexto para responder de manera precisa y completa.\n"
            f"Contexto: {context_text}\n"
            f"Pregunta: {question}\n"
            "Respuesta:"
        )

        #Generar texto
        output = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=50
        )[0]["generated_text"]

        #Post-procesado: tomar solo la parte que sigue a "Respuesta:"
        if "Respuesta:" in output:
            answer = output.split("Respuesta:")[-1].strip()
        else:
            answer = output.strip()

        return answer



#Ejemplo de uso:
if __name__ == "__main__":
    chatbot = AstroRag()
    pregunta = "What is an exoplanet?"
    respuesta = chatbot.rag_query(pregunta, max_new_tokens=30)
    print(respuesta)
