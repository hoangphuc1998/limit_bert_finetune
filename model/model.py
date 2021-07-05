import pytorch_lightning as pl
import datasets
import torch
from transformers import AutoModelForTokenClassification, get_cosine_schedule_with_warmup
import numpy as np

class QAModel(pl.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.model = AutoModelForTokenClassification.from_pretrained(config.model_type, num_labels=config.num_labels)
        self.tokenizer = tokenizer
        self.metric = datasets.load_metric('seqeval')
        self.freeze = False
        self.ner_map = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
    
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
        labels = batch["labels"]
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        self.log("train/loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        return loss


    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"].detach()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss, logits = outputs.loss, outputs.logits
        preds = torch.argmax(logits, dim=-1).detach()
        pred_labels = []
        gold_labels = []
        for pred, label in zip(preds, labels):
            mask = label != -100
            pred = pred[mask]
            label = label[mask]
            if len(label)>0:
                pred_labels.append(np.vectorize(self.ner_map)(pred.cpu().numpy()))
                gold_labels.append(np.vectorize(self.ner_map)(label.cpu().numpy()))
        self.metric.add_batch(predictions = pred_labels, references=gold_labels)
        self.log("val/loss", loss, prog_bar=False, logger=True, on_epoch=True, on_step=False)

    def validation_epoch_end(self, outputs):
        score = self.metric.compute()
        for entity in ['PER', 'ORG', 'LOC', 'MISC']:
            print(f"val/{entity}-F1: {score[entity]['f1']:.3f}", end=', ')
            self.log(f"val/{entity}-F1", score[entity]['f1'], logger=True)
        self.log("val/F1", score['overall_f1'], logger=True)
        print("\nF1: " + str(score['overall_f1']))
        self.metric = datasets.load_metric('seqeval')

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