from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn

from titanet.model.TitaNet import TitaNet

class LiveLossPlotCallback(pl.Callback):
    """
    Lightweight matplotlib plotter that keeps a running loss curve so you can
    monitor training/validation trajectories in real time and via a saved image.
    """

    def __init__(
        self,
        save_path: str = "lightning_logs/training_curve.png",
        show_live: bool = True,
    ):
        super().__init__()
        self.save_path = Path(save_path) if save_path else None
        self.show_live = show_live and plt.get_backend().lower() != "agg"
        self._epochs: list[int] = []
        self._train_losses: list[float | None] = []
        self._val_losses: list[float | None] = []
        self._fig = None
        self._ax = None
        self._announced = False

        if self.save_path:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def on_fit_start(self, trainer, pl_module):
        if self.show_live:
            plt.ion()

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch_idx = trainer.current_epoch + 1

        train_loss = self._get_metric(metrics, ("train_loss_epoch", "train_loss"))
        val_loss = self._get_metric(metrics, ("val_loss", "val_loss_epoch"))

        self._epochs.append(epoch_idx)
        self._train_losses.append(train_loss)
        self._val_losses.append(val_loss)

        self._update_plot(trainer)

    def on_fit_end(self, trainer, pl_module):
        if self.show_live and self._fig:
            plt.ioff()
            plt.show(block=False)

    def _get_metric(self, metrics, keys):
        for key in keys:
            if key not in metrics:
                continue
            value = metrics[key]
            if hasattr(value, "item"):
                try:
                    return float(value.item())
                except Exception:
                    pass
            try:
                return float(value)
            except Exception:
                continue
        return None

    def _update_plot(self, trainer):
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots()

        self._ax.clear()

        train_points = [(e, l) for e, l in zip(self._epochs, self._train_losses) if l is not None]
        val_points = [(e, l) for e, l in zip(self._epochs, self._val_losses) if l is not None]

        if train_points:
            epochs, losses = zip(*train_points)
            self._ax.plot(epochs, losses, label="train_loss", color="#1f77b4")
        if val_points:
            epochs, losses = zip(*val_points)
            self._ax.plot(epochs, losses, label="val_loss", color="#d62728")

        self._ax.set_xlabel("Epoch")
        self._ax.set_ylabel("Loss")
        self._ax.set_title("TitaNet training status")
        self._ax.legend()
        self._ax.grid(True, linestyle="--", alpha=0.3)
        self._fig.tight_layout()

        if self.show_live:
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            plt.pause(0.001)

        if self.save_path:
            self._fig.savefig(self.save_path, dpi=120)
            if not self._announced:
                trainer.print(f"[LivePlot] Saving training curve to {self.save_path}")
                self._announced = True
                
class TrainerModule(pl.LightningModule):
    def __init__(
        self,
        model: TitaNet | nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()

        base_model: nn.Module
        if isinstance(model, TitaNet):
            base_model = model.get_model()
        elif isinstance(model, nn.Module):
            base_model = model
        else:
            raise TypeError("TrainerModule expects a TitaNet wrapper or torch.nn.Module.")

        self.model = base_model
        self.model.train()  # ensure train mode even if loaded in eval for inference
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x, lengths=None):
        """
        Support both raw audio + lengths (preferred) and plain tensor input.
        """
        if lengths is not None:
            return self.model(input_signal=x, input_signal_length=lengths)
        return self.model(x)

    def _extract_logits(self, outputs):
        if isinstance(outputs, dict):
            if "logits" in outputs:
                return outputs["logits"]
            if "output" in outputs:
                return outputs["output"]
        if isinstance(outputs, (list, tuple)):
            return outputs[0]
        return outputs

    def on_fit_start(self):
        # Some pretrained NeMo modules are loaded in eval mode; force train to avoid frozen stats.
        self.model.train()

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            inputs, input_lengths, targets = batch
        else:
            inputs, targets = batch
            input_lengths = None

        outputs = self(inputs, input_lengths)
        logits = self._extract_logits(outputs)

        loss = self.criterion(logits, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=inputs.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            inputs, input_lengths, targets = batch
        else:
            inputs, targets = batch
            input_lengths = None

        outputs = self(inputs, input_lengths)
        logits = self._extract_logits(outputs)
        loss = self.criterion(logits, targets)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=inputs.size(0),
        )
        return loss

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("No trainable parameters found for optimizer.")
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

class TrainerWrapper:
    def __init__(
        self,
        model: pl.LightningModule,
        train_dataloader=None,
        val_dataloader=None,
        lr: float = 1e-3,
        accelerator: str = "gpu",
        devices: int = 1,
        max_epochs: int = 50,
        precision: str | int = "32-true",
        callbacks: list | None = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # You can still pass lr into the model if you want
        if lr is not None and hasattr(self.model, "lr"):
            self.model.lr = lr

        self.trainer = pl.Trainer(
            accelerator=accelerator,
            benchmark=True,    
            devices=devices,
            max_epochs=max_epochs,
            precision=precision,
            accumulate_grad_batches=4,         
            gradient_clip_val=1.0,
            callbacks=callbacks or [],
        )

    def fit(self):
        """Train the model."""
        self.trainer.fit(
            self.model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader,
        )

    def validate(self):
        """Run validation only."""
        return self.trainer.validate(
            self.model,
            dataloaders=self.val_dataloader,
        )

    def test(self, test_dataloader=None):
        """Run test."""
        return self.trainer.test(
            self.model,
            dataloaders=test_dataloader,
        )

    def save_checkpoint(self, path: str):
        self.trainer.save_checkpoint(path)
