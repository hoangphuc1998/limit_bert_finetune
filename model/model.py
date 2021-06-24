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
        self.metric = datasets.load_metric("squadv2")
    
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
        loss = outputs[0]
        self.log("train/loss", loss, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        start_scores, end_scores = self.model(input_ids=input_ids, attention_mask=attention_mask)
        answer_starts = torch.argmax(start_scores)
        answer_ends = torch.argmax(end_scores) + 1
        question_ids = batch["ids"]
        gold_answers = batch["answers"]
        pred_answers = []
        for question_id, input_id, answer_start, answer_end in zip(question_ids, input_ids, answer_starts, answer_ends):
            answer_text = self.tokenizer.decode(input_id[answer_start:answer_end])
            pred_answers.append({'prediction_text': answer_text, 'id': question_id, 'no_answer_probability': 0.0})
        self.metric.add_batch(predictions=pred_answers, references=gold_answers)

    def validation_epoch_end(self):
        score = self.metric.compute()
        self.log("val/EM", score["exact"], logger=True)
        self.log("val/F1", score["F1"], logger=True)
        self.metric = datasets.load_metric("squadv2")
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        return optimizer
