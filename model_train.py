import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from titanet.model.TitaNet import TitaNet
from titanet.training.titanet import TrainerModule, TrainerWrapper

def main():
    
    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints/",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    model = TitaNet(
        config_path="configs/lightweight_titanet.yaml",
        pretrained=False,
        device="cpu",
        use_pruned_model=False,
    )
    
    model = TrainerModule(model, lr=1e-3)
    model_trainer = TrainerWrapper(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_epochs=10,
        callbacks=[checkpoint_cb, lr_monitor],
    )