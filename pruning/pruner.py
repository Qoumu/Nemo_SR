
import torch
import torch.nn.utils.prune as prune
from pytorch_lightning.callbacks import Callback


class PruneCallback(Callback):
    """
    Periodically apply structured pruning during training and clean up masks at the end
    so checkpoints store only the pruned weights (no optimizer state, no masks).
    """

    def __init__(self, amount: float = 0.5, every_n_epochs: int = 2):
        super().__init__()
        self.amount = amount
        self.every_n_epochs = max(1, int(every_n_epochs))

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return

        for _, module in pl_module.model.named_modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
                prune.ln_structured(module, name="weight", amount=self.amount, n=2, dim=0)

    def on_train_end(self, trainer, pl_module):
        # Drop masks/reparametrizations so saved checkpoints stay compact.
        for _, module in pl_module.model.named_modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
                if hasattr(module, "weight_orig"):
                    prune.remove(module, "weight")
