import torch
import numpy as np
from .preprocess import *
import datasets

class HateXplainDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, split='train', max_length=128):
        super().__init__()
        data = datasets.load_dataset('hatexplain', split=split)
        self.tokens, self.labels, self.rationales = read_hatexplain(data)
        self.encodings = tokenizer(self.tokens, is_split_into_words=True, return_offsets_mapping=True, 
                                    truncation=True, padding=True, max_length=max_length)
        self.rationales = encode_tags(self.rationales, self.encodings)
        
    def __getitem__(self, idx):
        d = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        d['cls'] = self.labels[idx]
        d['rationale'] = self.rationales[idx]
        return d

    def __len__(self):
        return len(self.tokens)