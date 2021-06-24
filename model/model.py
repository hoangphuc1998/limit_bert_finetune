import pytorch_lightning as pl
import datasets
import torch
from transformers import AutoModelForQuestionAnswering

class QAModel(pl.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.model = AutoModelForQuestionAnswering.from_pretrained(config.model_type)
        self.tokenizer = tokenizer
        self.metric = datasets.load_metric("squad_v2")
    
    def forward(self, contexts, questions):
        '''
        Not implemented yet
        '''
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        start_positions = batch["start_positions"]
        end_positions = batch["end_positions"]
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss
        self.log("train/loss", loss, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        start_scores, end_scores = outputs.start_logits, outputs.end_logits
        answer_starts = torch.argmax(start_scores, dim=-1)
        answer_ends = torch.argmax(end_scores, dim=-1) + 1
        question_ids = batch["id"]
        gold_answers = []
        pred_answers = []
        for question_id, input_id, answer_start, answer_end, gold_answer in zip(question_ids, input_ids, answer_starts, answer_ends, batch["answers"]):
            answer_text = self.tokenizer.decode(input_id[answer_start:answer_end])
            pred_answers.append({'prediction_text': answer_text, 'id': question_id, 'no_answer_probability': 0.0})
            gold_answers.append({'answers': gold_answer, 'id': question_id})
        return gold_answers, pred_answers

    def validation_epoch_end(self, outputs):
        gold_answers, pred_answers = zip(*outputs)
        gold_answers = gold_answers[0]
        pred_answers = pred_answers[0]
        score = self.metric.compute(predictions=pred_answers, references=gold_answers)
        self.log("val/EM", score["exact"], logger=True)
        self.log("val/F1", score["f1"], logger=True)
        self.metric = datasets.load_metric("squad_v2")
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        return optimizer
