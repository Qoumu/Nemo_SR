
import torch
import torch.nn.utils.prune as prune
from pytorch_lightning.callbacks import Callback

class PruneCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print("DEBUGG: Pruning Callback Triggered")
        epoch = trainer.current_epoch
        if epoch%2 == 0:
            for name, module in pl_module.model.named_modules():
                if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
                    prune.l1_unstructured(module, name="weight", amount=0.3)
            print("Applied pruning at epoch", epoch)