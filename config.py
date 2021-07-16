import argparse

class HateXplainConfig():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # Model
        self.parser.add_argument("--model_type", type=str, default="bert-large-uncased", help="Model type")
        self.parser.add_argument("--num_classes", type=int, default=3, help="Number of NER tags")
        self.parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
        self.parser.add_argument("--dropout", type=float, default=0.1, help="Dropout of hidden layer in bert")

        # Training
        self.parser.add_argument("--seed", type=int, default=42, help="Random seed number")
        self.parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
        self.parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
        self.parser.add_argument("--bert_lr", type=float, default=5e-5, help="Learning rate for bert model")
        self.parser.add_argument("--bz", type=int, default=16, help="Batch size")
        self.parser.add_argument("--freeze_steps", type=int, default=500, help="Freeze main model to train qa head")
        self.parser.add_argument("--num_warmup_steps", type=int, default=1500, help="Number of warmup steps")
        self.parser.add_argument("--grad_clip", type=float, default=2.0, help="Gradient clipping value")
        self.parser.add_argument("--alpha", type=float, default=1.0, help="Weight between classification and rationale loss")
        self.parser.add_argument("--num_workers", type=int, default=0, help="Number of workers")

        # Save folder
        self.parser.add_argument("--save_folder", type=str, default="/content/drive/MyDrive/NLP/")
        self.parser.add_argument("--project", type=str, default="hatexplain", help="Wandb project name")
        self.parser.add_argument("--exp", type=str, default="bert-large-exp1", help="Wandb experiment name")

    def parse_args(self):
        return self.parser.parse_known_args()[0]