
import torch
from torch.utils.data import Dataset

#Crear un Dataset tipo pytorch
class ChatDataset(Dataset):
    def __init__(self, encodings):
        #encodings es el output del tokenizer (input_ids, attention_mask)
        self.encodings = encodings  

    def __len__(self):
        return len(self.encodings["input_ids"])  

    def __getitem__(self, idx):
        #GPT2 necesita input_ids y labels iguales
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} #en key estan todo el texto encodeadas
        item["labels"] = item["input_ids"].clone()  
        return item




