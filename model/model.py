import pytorch_lightning as pl
import datasets
import torch
from transformers import get_cosine_schedule_with_warmup
import numpy as np
from torchmetrics import F1
from .bert import BertModelForHateSpeech

class HateXplainModel(pl.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.model = BertModelForHateSpeech(config)
        self.tokenizer = tokenizer
        self.cls_metric = F1(num_classes=config.num_classes)
        self.rationale_metric = F1(num_classes=2)
        self.freeze = False
    
    def setup(self, stage):
        if stage == 'fit':
            train_batches = len(self.train_dataloader())
            self.train_steps = (self.config.epochs * train_batches)
    
    def forward(self, sentence):
        encoding = self.tokenizer(sentence.split(), is_split_into_words=True, return_offsets_mapping=True, return_tensors='pt')
        output = self.model(encoding.input_ids, encoding.attention_mask)
        cls_logits = output['cls_logits']
        rationale_logits = output['rationale_logits']
        cls_label = torch.argmax(cls_logits, dim=-1)[0].item()
        rationale_labels = torch.argmax(rationale_logits[0], dim=-1)
        offset_mapping = encoding.offset_mapping[0]
        input_ids = encoding.input_ids[0]
        rationale_list = []
        rationale=[]
        is_continuous=False
        i=1
        print(rationale_labels)
        while i<=len(input_ids)-1:
            if (rationale_labels[i]==1 and offset_mapping[i][0]==0) or (offset_mapping[i][0]!=0 and is_continuous):
                is_continuous=True
                rationale.append(input_ids[i])
            elif len(rationale)>0:
                is_continuous=False
                rationale_list.append(self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(rationale)))
                rationale=[]
            i+=1
        if len(rationale)>0:
            rationale_list.append(self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(rationale)))
        return cls_label, rationale_list

    def training_step(self, batch, batch_idx):
        if self.global_step < self.config.freeze_steps:
            self.freeze_model()
        else:
            self.unfreeze_model()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        cls_labels = batch["cls"]
        rationale_labels = batch["rationale"]
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, cls_labels=cls_labels, rationale_labels=rationale_labels)
        loss = outputs['loss']
        self.log("train/loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        cls_labels = batch["cls"].detach()
        rationale_labels = batch["rationale"].detach()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, cls_labels=cls_labels, rationale_labels=rationale_labels)
        loss, cls_logits, rationale_logits = outputs['loss'], outputs['cls_logits'], outputs['rationale_logits']
        cls_preds = torch.argmax(cls_logits, dim=-1).detach()
        rationale_preds = torch.argmax(rationale_logits, dim=-1).detach()

        mask = rationale_labels.view(-1)!=-100
        rationale_preds = rationale_preds.view(-1)[mask]
        rationale_labels = rationale_labels.view(-1)[mask]

        # self.metric.add_batch(predictions = pred_labels, references=gold_labels)
        self.cls_metric(cls_preds, cls_labels)
        self.rationale_metric(rationale_preds, rationale_labels)
        self.log("val/cls_F1", self.cls_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/rationale_F1", self.rationale_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/loss", loss, prog_bar=False, logger=True, on_epoch=True, on_step=False)

    def validation_epoch_end(self, outputs):
        print(f"Cls F1: {self.cls_metric.compute().item():.3f}, Rationale F1: {self.rationale_metric.compute().item():.3f}")

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        cls_labels = batch["cls"].detach()
        rationale_labels = batch["rationale"].detach()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, cls_labels=cls_labels, rationale_labels=rationale_labels)
        loss, cls_logits, rationale_logits = outputs['loss'], outputs['cls_logits'], outputs['rationale_logits']
        cls_preds = torch.argmax(cls_logits, dim=-1).detach()
        rationale_preds = torch.argmax(rationale_logits, dim=-1).detach()

        mask = rationale_labels.view(-1)!=-100
        rationale_preds = rationale_preds.view(-1)[mask]
        rationale_labels = rationale_labels.view(-1)[mask]

        # self.metric.add_batch(predictions = pred_labels, references=gold_labels)
        self.cls_metric(cls_preds, cls_labels)
        self.rationale_metric(rationale_preds, rationale_labels)

    def test_epoch_end(self, outputs):
        print(f"Cls F1: {self.cls_metric.compute().item():.3f}, Rationale F1: {self.rationale_metric.compute().item():.3f}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([{'params': layer.parameters(), 'lr':self.config.bert_lr if 'bert' in name else self.config.lr}
                                        for name, layer in self.model.named_children()], weight_decay=1e-5)
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.config.num_warmup_steps, num_training_steps=self.train_steps)
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

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