import json
import re
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

#Abri json y cargar
with open("./results/astronomia_corpus.json", "r", encoding="utf-8") as f: #Asegurarse estar en el directorio correcto
    docs = json.load(f)

#Obtener solo el texto
texts = [d["text"] for d in docs]

#Dividr los datos 80% train y 20% val
train_texts, val_texts = train_test_split(texts, test_size=0.2, random_state=RANDOM_STATE)

print(len(train_texts), len(val_texts))


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # quitar URLs
    text = re.sub(r"[^a-z0-9\s.,;:!?()\-']", " ", text)  # quitar rarezas
    text = re.sub(r"\s+", " ", text).strip()
    return text

train_texts = [clean_text(t) for t in train_texts]
eval_texts = [clean_text(t) for t in val_texts]

#Crear un Dataset tipo pytorch
class ChatDataset(Dataset):
    def __init__(self, text, tokenizer, block_size=128):
        self.examples = tokenizer(
            text,
            truncation = True,
            padding = "max_length",
            max_length = block_size,
            return_tensors = "pt"
        )["input_ids"]

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        return self.examples[index]

#Definir el trabformador de GPT2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  #evitar errores

#Crear Dataset 

train_dataset = ChatDataset(train_texts, tokenizer)

print(next(iter(train_dataset)))


