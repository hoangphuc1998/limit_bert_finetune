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
        self.parser.add_argument("--seed", type=int, default=42, help="Random seed number")
        self.parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
        self.parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
        self.parser.add_argument("--bz", type=int, default=16, help="Batch size")

        # Save folder
        self.parser.add_argument("--save_folder", type=str, default="/content/drive/MyDrive/NLP/")
        self.parser.add_argument("--project", type=str, default="squad_qa", help="Wandb project name")
        self.parser.add_argument("--exp", type=str, default="bert-base-exp0", help="Wandb experiment name")

    def parse_args(self):
        return self.parser.parse_knowns_args()