import pytorch_lightning as pl
import torch
import torch.nn as nn

from titanet.model.TitaNet import TitaNet


class TrainerModule(pl.LightningModule):
    def __init__(self, model: TitaNet | nn.Module, lr: float = 1e-3):
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
        optimizer = torch.optim.Adam(params, lr=self.lr)
        return optimizer
    
class TrainerWrapper:
    def __init__(
        self,
        model: pl.LightningModule,
        train_dataloader=None,
        val_dataloader=None,
        lr: float = 1e-3,
        accelerator: str = "auto",
        devices: int = 1,
        max_epochs: int = 10,
        precision: str | int = "32-true",
        callbacks: list | None = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # You can still pass lr into the model if you want
        if hasattr(self.model, "lr"):
            self.model.lr = lr

        self.trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            precision=precision,
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
