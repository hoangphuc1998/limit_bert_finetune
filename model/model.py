import pytorch_lightning as pl
import datasets
import torch
from transformers import AutoModelForTokenClassification
import torchmetrics

class QAModel(pl.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.model = AutoModelForTokenClassification.from_pretrained(config.model_type, num_labels=config.num_labels)
        self.tokenizer = tokenizer
        self.metric = torchmetrics.F1(num_classes=config.num_labels)
        self.freeze = False
    
    def forward(self, contexts, questions):
        '''
        Not implemented yet
        '''
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        if self.global_step < self.config.freeze_steps:
            self.freeze_model()
        else:
            self.unfreeze_model()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        self.log("train/loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        return loss


    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss, logits = outputs.loss, outputs.logits
        preds = torch.argmax(logits, dim=-1)
        self.metric.update(preds, labels)
        self.log("val/loss", loss, prog_bar=False, logger=True, on_epoch=True, on_step=False)

    def validation_epoch_end(self, outputs):
        score = self.metric.compute()
        self.log("val/F1", score, logger=True)
        print("F1: " + str(score))
        self.metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        return optimizer

    def freeze_model(self):
        if self.freeze == False:
            for name, child in self.model.named_children():
                if name == "bert":
                    for param in child.parameters():
                        param.requires_grad = False
            self.freeze = True

    def unfreeze_model(self):
        if self.freeze == True:
            for param in self.model.parameters():
                param.requires_grad = True
            self.freeze = False