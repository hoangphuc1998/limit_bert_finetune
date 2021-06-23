import pytorch_lightning as pl
from transformers import AutoTokenizer
from .dataset import SquadDataset
from torch.utils.data import DataLoader

class SquadDataModule(pl.LightningDataModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

    def setup(self):
        self.train_dataset = SquadDataset(self.config.train_file, self.tokenizer)
        self.val_dataset = SquadDataset(self.config.val_file, self.tokenizer)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.bz, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=64, shuffle=False)