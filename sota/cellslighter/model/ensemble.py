import torch.nn as nn
import lightning as L
from torchvision import models
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, Precision, AUROC
import torch
from icecream import ic
from model.model import CellSlighter

class EnsembleClassifier(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.models: nn.ModuleList = kwargs.get('models', None)
        self.model_params = kwargs.get('model_params', None)
        
        if self.model_params is None and self.models is None:
            model_params = [{15}] * 10
            self.models = nn.ModuleList([CellSlighter(15) for i in range(10)])
        elif self.models is None and self.model_params is not None:
            self.models = nn.ModuleList([CellSlighter(**params) for params in self.model_params])
        else:
            pass
        self.accuracy_overall: MulticlassAccuracy = MulticlassAccuracy(
            num_classes=self.classes_num
        )
        self.f1_macro_overall: MulticlassF1Score = MulticlassF1Score(
            num_classes=self.classes_num,
            average='macro'
        )
        self.precision_cell_avg: Precision = Precision(
            'multiclass',
            num_classes=self.classes_num,
            average='macro'
        )
        # TODO:
        self.auroc_cell: AUROC = AUROC(
            'multiclass',
            num_classes=self.classes_num,
            average='macro')
    
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=True):
            outputs = [model(x) for model in self.models]
            
        return self.mode_logits(outputs)
    def voting(self, outputs):
        preds = self.mode_mps(outputs)
        return preds
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        optimizers = self.optimizers()
        
        for optimizer_idx, (model, optimizer) in enumerate(zip(self.models, optimizers)):
            optimizer.zero_grad()
            y_hat = model(x)
            loss = nn.functional.cross_entropy(y_hat, y)
            optimizer.step()
            
            self.log_dict({
                f"model_{optimizer_idx}_loss_train": loss.item(),
                f"model_{optimizer_idx}_accuracy_train": self.accuracy_overall(y_hat, y),
                f"model_{optimizer_idx}_f1_macro_train": self.f1_macro_overall(y_hat, y),
                f"model_{optimizer_idx}_precision_cell_avg_train": self.precision_cell_avg(y_hat, y),
                f"model_{optimizer_idx}_auroc_cell_train": self.auroc_cell(y_hat, y)
            }, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        model_losses = []
        for i, model in enumerate(self.models):
            y_hat = model(x)
            loss = nn.functional.cross_entropy(y_hat, y)
            
            self.log_dict({
                f"model_{i}_loss_val": loss.item(),
                f"model_{i}_accuracy_val": self.accuracy_overall(y_hat, y),
                f"model_{i}_f1_macro_val": self.f1_macro_overall(y_hat, y),
                f"model_{i}_precision_cell_avg_val": self.precision_cell_avg(y_hat, y),
                f"model_{i}_auroc_cell_val": self.auroc_cell(y_hat, y)
            }, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        
        avg_loss = torch.stack(model_losses).mean()
        y_hat_all = self(x)
        self.log_dict({
            "loss_val": avg_loss,
            "accuracy_val": self.accuracy_overall(y_hat_all, y),
            "f1_macro_val": self.f1_macro_overall(y_hat_all, y),
            "precision_cell_avg_val": self.precision_cell_avg(y_hat_all, y),
            "auroc_cell_val": self.auroc_cell(y_hat_all, y)
        }, logger=True, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizers = [torch.optim.Adam(model.parameters(), lr=self.model_params[i]['lr'], weight_decay=self.model_params[i]['weight_decay']) for i, model in enumerate(self.models)]
        return optimizers
    
    def mode_mps(self, votes):
        if votes.numel() == 0:
            return torch.tensor(0, device=votes.device)
        unique_values, counts = torch.unique(votes, return_counts=True)
        max_count_index = torch.argmax(counts)
        mode_value = unique_values[max_count_index]
        return mode_value
    
    def mode_logits(self, votes):
        if votes.numel() == 0:
            return torch.tensor(0, device=votes.device), torch.tensor([], device=votes.device)

        unique_values, counts = torch.unique(votes, return_counts=True)
        logits = counts.float() / votes.numel()
        return logits
