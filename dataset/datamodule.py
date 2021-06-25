from dataset.utils import my_collate
import pytorch_lightning as pl
from transformers import AutoTokenizer
from .dataset import SquadDataset
from torch.utils.data import DataLoader
from .utils import my_collate

class SquadDataModule(pl.LightningDataModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

    def setup(self, stage):
        self.train_dataset = SquadDataset(self.config.train_file, self.tokenizer, training=True, max_length=self.config.max_length)
        self.val_dataset = SquadDataset(self.config.val_file, self.tokenizer, training=False, max_length=self.config.max_length)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.bz, shuffle=True, collate_fn=my_collate)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=64, shuffle=False, collate_fn=my_collate)