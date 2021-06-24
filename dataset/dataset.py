import torch
from .preprocess import *

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, tokenizer, training=True):
        super().__init__()
        self.training = training
        self.contexts, self.questions, self.answers, self.ids = read_squad(json_file, training)
        self.encodings = tokenizer(self.contexts, self.questions, truncation=True, padding=True)
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