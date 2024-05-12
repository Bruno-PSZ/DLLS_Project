import torch.nn as nn
import lightning as L
from torchvision import models
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, Precision, AUROC
import torch
from icecream import ic
import timm


class CellSlighter(L.LightningModule):
    def __init__(self,
                 classes_num: int, input_len: int = 42, img_size: int | tuple[int, int] = 30,
                 dropouts: dict[str, list[float]] | None = None,
                 heads_architecture: dict[str, list[int]] | None = None, learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5, backbone: str = 'resnet50',
                 backbone_trainable_layers: int = 1
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate: float = learning_rate
        self.weight_decay: float = weight_decay
        self.classes_num: int = classes_num
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
            average='macro'
        )
        # TODO:
        self.pool: nn.Module = nn.AdaptiveAvgPool2d(1)
        if dropouts is not None and heads_architecture is not None:
            self.backbone: nn.Module = getattr(models, backbone)(num_classes=classes_num)
            last_channel: int = self.backbone.fc.in_features
            self.heads: nn.ModuleDict = nn.ModuleDict({
                name: self._make_classifier(
                    [last_channel, *layers, classes_num], dropouts[name]) for name, layers in heads_architecture.items()
            })
            # TODO:
        elif backbone == 'resnet50':
            self.backbone: nn.Module = getattr(models, backbone)(num_classes=classes_num)
            self.heads: nn.Module = nn.Identity()
            self.backbone.conv1 = nn.Conv2d(input_len, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                            bias=False)
        elif backbone == 'resnet50.a1_in1k':
            model = timm.create_model('resnet50.a1_in1k', pretrained=True)
        # for param in self.backbone[:-backbone_trainable_layers].parameters():
        #    param.requires_grad = False
        elif backbone == 'vit':
            self.backbone = timm.create_model('vit_tiny_patch16_224.augreg_in21k_ft_in1k', pretrained=False,
                                              img_size=img_size, in_chans=input_len, num_classes=self.classes_num)
            self.heads: nn.Module = nn.Identity()
    
    def _make_classifier(self, layers, dropouts):
        # print(layers)
        # print(list(zip(layers[:], layers[1:], dropouts[:-1])))
        return nn.Sequential(
            *[self._make_layer(in_features, out_features, dropout) for in_features, out_features, dropout in
              zip(layers[:-2], layers[1:-1], dropouts[:-1])],
            nn.Dropout(p=dropouts[-1]), nn.Linear(layers[-2], layers[-1]))
    
    def _make_layer(self, in_features, out_features, dropout, activation=nn.ReLU):
        return nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            activation(),
        )
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        with torch.cuda.amp.autocast(enabled=True):
            x = self.backbone(x)
            # ic(x.shape)
            # x = self.pool(x)
            # ic("after pool", x.shape)
            x = self.heads(x)
            # ic(x.shape)#TODO: check if nn.ModuleDict can be used here
        return x
    
    def training_step(self, batch: tuple[torch.tensor, torch.tensor], batch_idx: int) -> torch.tensor:
        x, y = batch
        # ic(x.shape)
        # ic(y)
        y_hat: torch.tensor = self(x)
        # ic(y_hat.shape)
        # ic(y_hat)
        loss: torch.tensor = nn.functional.cross_entropy(y_hat, y)
        # ic(loss)
        self.log_dict({
            "loss_train": loss,
            "accuracy_train": self.accuracy_overall(y_hat, y),
            "f1_macro_train": self.f1_macro_overall(y_hat, y),
            "precision_cell_avg_train": self.precision_cell_avg(y_hat, y),
            "auroc_cell_train": self.auroc_cell(y_hat, y)
        }, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch: tuple[torch.tensor, torch.tensor], batch_idx: int) -> None:
        x, y = batch
        # ic(x.shape)
        # ic(y)
        y_hat: torch.tensor = self(x)
        # ic(y_hat.shape)
        # ic(y_hat)
        loss: torch.tensor = nn.functional.cross_entropy(y_hat, y)
        # ic(loss)
        self.log_dict({
            "loss_val": loss,
            "accuracy_val": self.accuracy_overall(y_hat, y),
            "f1_macro_val": self.f1_macro_overall(y_hat, y),
            "precision_cell_avg_val": self.precision_cell_avg(y_hat, y),
            "auroc_cell_val": self.auroc_cell(y_hat, y)
        }, logger=True, on_step=True, on_epoch=True, sync_dist=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate,
                                weight_decay=self.hparams.weight_decay)
