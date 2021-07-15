from model import HateXplainModel
from dataset import HateXplainDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from config import HateXplainConfig
import pytorch_lightning as pl
import os
import json

if __name__ == "__main__":
    config = HateXplainConfig().parse_args()
    pl.seed_everything(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_type)
    save_path = os.path.join(config.save_folder, config.project, config.exp)
    # Save config file
    os.makedirs(save_path, exist_ok=True)
    json.dump(vars(config), open(os.path.join(save_path, "config.json"), "w"))
    # Callbacks
    model_checkpoint = ModelCheckpoint(
        dirpath=save_path, monitor="val/cls_F1", mode="max", save_last=True)
    lr_logging = LearningRateMonitor(logging_interval="step")
    wandb_logger = WandbLogger(
        project=config.project, name=config.exp, config=vars(config))

    # Training
    hatexplain_dm = HateXplainDataModule(config, tokenizer)
    model = HateXplainModel(config, tokenizer)
    trainer = pl.Trainer(callbacks=[model_checkpoint, lr_logging], logger=wandb_logger, gpus=-1, check_val_every_n_epoch=1,
                         gradient_clip_val=config.grad_clip, track_grad_norm=2, max_epochs=config.epochs, limit_train_batches=100)
    trainer.fit(model, hatexplain_dm)
