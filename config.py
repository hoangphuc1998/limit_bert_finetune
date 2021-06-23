import argparse

class QAConfig():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # Dataset
        self.parser.add_argument("--train_file", type=str, default="/content/squad/train-v2.0.json", help="Train json file")
        self.parser.add_argument("--val_file", type=str, default="/content/squad/dev-v2.0.json", help="Validation json file")

        # Model
        self.parser.add_argument("--model_type", type=str, default="bert-large-uncased", help="Model type")

        # Training
        self.parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")

    def parse_args(self):
        return self.parser.parse_knowns_args()