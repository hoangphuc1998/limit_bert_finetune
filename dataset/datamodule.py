import pytorch_lightning as pl
from transformers import AutoTokenizer
from .dataset import SquadDataset
from torch.utils.data import DataLoader

class SquadDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model)

        self.train_dataset = SquadDataset(self.config.train_file, tokenizer)
        self.val_dataset = SquadDataset(self.config.val_file, tokenizer)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.bz, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=64, shuffle=False)