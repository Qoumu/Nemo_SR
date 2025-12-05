import pytorch_lightning as pl
import torch
import torch.nn.utils.prune as prune

from titanet.model.TitaNet import TitaNet

class TrainerModule(pl.LightningModule):
    def __init__(self, model: TitaNet, lr: float = 1e-3,):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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
        max_epochs: int = 10,
        precision: str | int = "16-mixed",
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