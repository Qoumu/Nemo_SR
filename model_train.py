import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from titanet.model.TitaNet import TitaNet
from titanet.training.titanet import TrainerModule, TrainerWrapper
from pruning.pruner import PruneCallback
from data.speakerdataset.dataloader import Dataloader, collate_fn

def main():
    
    # Prepare Dataset
    train_dataset = Dataloader(manifest_path="data/speakers/known/speaker.json")
    val_dataset = Dataloader(manifest_path="data/speakers/unknown/speaker.json")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )
    
    prune_cp = PruneCallback()
    
    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints/",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    callbacks=[prune_cp, checkpoint_cb, lr_monitor]

    model = TitaNet(
        config_path="configs/lightweight_titanet.yaml",
        device="cuda",
        use_pruned_model=False,
    )
    
    model = TrainerModule(model, lr=1e-3)
    model_trainer = TrainerWrapper(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_epochs=10,
        callbacks=callbacks
    )
    
    model_trainer.fit()
    
if __name__ == "__main__":
    main()