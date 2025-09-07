#Crear mediante Gradio una pqueña interfaza, donde se pida una pregunta del usuario y genre una respuesta de ambos modelos

from rag import AstroRag
import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import warnings
import torch

warnings.filterwarnings("ignore")
MODEL_DIR = "./gpt2-astrotuned"
DEVICE = 0 if torch.cuda.is_available() else -1

# Cargar modelo GPT-2 simple

model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)

# Inicializar modelo + RAG
rag = AstroRag()

# Funciones de generación
def generate_gpt2(prompt, top_k, max_new_tokens, temperature):
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device= DEVICE
    )
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=top_k,
        temperature=temperature
    )
    return output[0]["generated_text"]

def generate_rag(question, top_k, max_context_tokens, max_new_tokens, temperature):
    # Llamar a la función del RAG que ya tienes en AstroRag
    # Aquí puedes modificarla para que acepte parámetros dinámicos
    return rag.rag_query(
        question,
        top_k=top_k,
        max_context_tokens=max_context_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )

# Función unificada para Gradio
def chat(question, model_type, top_k, max_context_tokens, max_new_tokens, temperature):
    if model_type == "GPT-2":
        return generate_gpt2(question, top_k, max_new_tokens, temperature)
    elif model_type == "GPT-2 + RAG":
        return generate_rag(question, top_k, max_context_tokens, max_new_tokens, temperature)


# Interfaz Gradio
with gr.Blocks() as demo:
    gr.Markdown("### Chatbot Astronómico")
    gr.Markdown("**Explicación de parámetros:**\n"
                "- **top_k**: número de tokens candidatos que el modelo considera.\n"
                "- **max_context_tokens**: tokens máximos tomados del contexto (solo RAG).\n"
                "- **max_new_tokens**: tokens máximos a generar en la respuesta.\n"
                "- **temperature**: aleatoriedad de la respuesta (0=determinista, >0=creativo).")
    
    question_input = gr.Textbox(label="Pregunta")
    model_select = gr.Dropdown(["GPT-2", "GPT-2 + RAG"], label="Modelo")
    top_k_slider = gr.Slider(1, 100, value=50, step=1, label="top_k")
    max_context_slider = gr.Slider(10, 500, value=200, step=10, label="max_context_tokens")
    max_new_slider = gr.Slider(10, 500, value=50, step=5, label="max_new_tokens")
    temperature_slider = gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="temperature")
    output_box = gr.Textbox(label="Respuesta")
    
    btn = gr.Button("Generar")
    btn.click(
        chat,
        inputs=[question_input, model_select, top_k_slider, max_context_slider, max_new_slider, temperature_slider],
        outputs=output_box
    )

demo.launch()
