import torch
import numpy as np
from .preprocess import *
import datasets

class CoNLLDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, split="train", max_length=128):
        self.tokenizer = tokenizer
        conll_dataset = datasets.load_dataset("conll2003", split=split)
        self.num_rows = conll_dataset.num_rows
        self.encodings = tokenizer(conll_dataset["tokens"], is_split_into_words=True, return_offsets_mapping=True, 
                                    truncation=True, padding=True, max_length=max_length)
        self.labels = encode_tags(conll_dataset["ner_tags"], self.encodings)
        self.encodings.pop("offset_mapping")
    
    def __getitem__(self, idx):
        d = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        d["labels"] = self.labels[idx]
        return d

    def __len__(self):
        return self.num_rows