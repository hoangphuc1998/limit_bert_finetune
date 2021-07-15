from transformers import AutoModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch

class BertModelForHateSpeech(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.model_type)
        self.classifier = nn.Linear(self.bert.hidden_size, config.num_classes)
        self.rationale_head = nn.Linear(self.bert.hidden_size, 2)
        self.dropout = nn.Dropout(config.dropout)
        self.init_weights()
    
    def forward(self, input_ids, attention_mask, cls_labels=None, rationale_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask= attention_mask)
        last_outputs = outputs[0]
        last_outputs = self.dropout(last_outputs)
        cls_token_output = last_outputs[:,0,:]
        rationale_logits = self.rationale_head(last_outputs)
        cls_logits = self.classifier(cls_token_output)
        loss = None
        if cls_labels is not None and rationale_labels is not None:
            cls_criterion = CrossEntropyLoss()
            rationale_criterion = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = rationale_logits.view(-1, 2)
                active_labels = torch.where(
                    active_loss, rationale_labels.view(-1), torch.tensor(rationale_criterion.ignore_index).type_as(rationale_labels)
                )
                rationale_loss = rationale_criterion(active_logits, active_labels)
            else:
                rationale_loss = rationale_criterion(rationale_logits.view(-1, self.num_labels), rationale_labels.view(-1))
            cls_loss = cls_criterion(cls_logits, cls_labels)
            loss = cls_loss + self.config.alpha * rationale_loss
        d = {"cls_logits": cls_logits, "rationale_logits": rationale_logits, "loss": loss}
        return d

    def init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()
        self.rationale_head.weight.data.normal_(mean=0.0, std=0.02)
        self.rationale_head.bias.data.zero_()
