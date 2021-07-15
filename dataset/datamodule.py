import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .dataset import HateXplainDataset

class HateXplainDataModule(pl.LightningDataModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

    def setup(self, stage):
        if stage=="fit":
            self.train_dataset = HateXplainDataset(self.tokenizer, split="train", max_length=self.config.max_length)
            self.val_dataset = HateXplainDataset(self.tokenizer, split="validation", max_length=self.config.max_length)
        else:
            self.test_dataset = HateXplainDataset(self.tokenizer, split="test", max_length=self.config.max_length)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.bz, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=64, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=64, shuffle=False)