import torch
from .preprocess import *

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, tokenizer):
        super().__init__()
        contexts, questions, answers = read_squad(json_file)
        add_end_idx(answers, contexts)
        self.encodings = tokenizer(contexts, questions, trucation=True, padding=True)
        add_token_positions(self.encodings, answers, tokenizer)
    
    def __getitem__(self, idx):
        return {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings["input_ids"])