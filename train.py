from model import QAModel
from dataset import SquadDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from config import QAConfig
import pytorch_lightning as pl
import os
import json

if __name__ == "__main__":
    config = QAConfig().parse_args()
    pl.seed_everything(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_type)
    save_path = os.path.join(config.save_folder, config.project, config.exp)
    # Save config file
    os.makedirs(save_path, exist_ok=True)
    json.dump(vars(config), open(os.path.join(save_path, "config.json"), "w"))
    # Callbacks
    model_checkpoint = ModelCheckpoint(dirpath=save_path, monitor="val/F1", mode="max", save_last=True)
    lr_logging = LearningRateMonitor(logging_interval="step")
    wandb_logger = WandbLogger(project=config.project, name=config.exp, config=vars(config))

    # Training
    squad_dm = SquadDataModule(config, tokenizer)
    qa_model = QAModel(config, tokenizer)
    trainer = pl.Trainer(callbacks=[model_checkpoint, lr_logging], logger=wandb_logger, gpus=-1, check_val_every_n_epoch=1)
    trainer.fit(qa_model, squad_dm)