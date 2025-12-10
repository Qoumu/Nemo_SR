import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pathlib import Path

from titanet.model.TitaNet import TitaNet
from titanet.training.titanet import LiveLossPlotCallback, TrainerModule, TrainerWrapper
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
    
    prune_cp = PruneCallback(amount=0.5, every_n_epochs=2)
    
    checkpoint_cb = ModelCheckpoint(
        dirpath="ckpts/",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    plot_cb = LiveLossPlotCallback(
        save_path="lightning_logs/training_curve.png",
        show_live=True,
    )
    
    callbacks=[ # prune_cp,
               checkpoint_cb,
               lr_monitor,
               plot_cb]

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

    # Final compact checkpoint with only model weights (no optimizer state, no masks).
    Path("ckpts").mkdir(parents=True, exist_ok=True)
    model_trainer.trainer.save_checkpoint("ckpts/pruned.ckpt", weights_only=True)

    # Export to NeMo archive for deployment.
    export_dir = Path("export")
    export_dir.mkdir(parents=True, exist_ok=True)
    nemo_path = export_dir / "titanet_finetuned.nemo"
    base_model = getattr(model_trainer.model, "model", None)
    if base_model is not None:
        base_model.to("cpu")
        base_model.eval()
        base_model.save_to(str(nemo_path))
        print(f"Exported NeMo model to {nemo_path}")
    
if __name__ == "__main__":
    main()
