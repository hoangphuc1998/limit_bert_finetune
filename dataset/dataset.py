import torch
import numpy as np
from .preprocess import *
import datasets

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, tokenizer, training=True, max_length=128):
        super().__init__()
        self.training = training
        self.contexts, self.questions, self.answers, self.ids = read_squad(json_file, training)
        self.encodings = tokenizer(self.contexts, self.questions, truncation=True, padding=True, max_length=max_length)
        if training:
            add_end_idx(self.answers, self.contexts)
            add_token_positions(self.encodings, self.answers, tokenizer)
    
    def __getitem__(self, idx):
        d = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        if not self.training:
            d["id"] = self.ids[idx]
            d['answers'] = self.answers[idx]
        return d

    def __len__(self):
        return len(self.encodings["input_ids"])

class CoNLLDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, split="train", max_length=128):
        self.tokenizer = tokenizer
        self.dataset = datasets.load_dataset("conll2003", split)
        self.encodings = tokenizer(self.dataset.tokens, is_split_into_words=True, return_offsets_mapping=True, 
                                    truncation=True, padding=True, max_length=max_length)
        self.labels = encode_tags(self.dataset.ner_tags, self.encodings)
    
    def __getitem__(self, idx):
        d = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        d["labels"] = self.labels[idx]
        return d

    def __len__(self):
        return self.dataset.num_rows