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


    # def validation_epoch_end(self, outputs):
    #     pred_labels = []
    #     gold_labels = []
    #     for output in outputs:
    #         pred_labels+=output["pred_labels"]
    #         gold_labels+=output["gold_labels"]
    #     score = self.metric.compute(predictions = pred_labels, references=gold_labels)
    #     for entity in ['PER', 'ORG', 'LOC', 'MISC']:
    #         print(f"val/{entity}-F1: {score[entity]['f1']:.3f}", end=', ')
    #         self.log(f"val/{entity}-F1", score[entity]['f1'], logger=True)
    #     self.log("val/F1", score['overall_f1'], logger=True)
    #     print("\nF1: " + str(score['overall_f1']))

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