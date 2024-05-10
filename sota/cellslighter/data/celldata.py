from typing import List

import numpy as np
from torchvision.datasets import VisionDataset

from data.cellcrop import CellCrop
import torch


class CellDataset(VisionDataset):
    def __init__(self, crops: List[CellCrop], transform=None) -> None:
        super().__init__()
        self.crops = crops
        self.transform = transform
    
    def __len__(self):
        return len(self.crops)
    
    def __getitem__(self, item):
        sample = self.crops[item].sample()
        aug = np.dstack(
            [sample['mask'][:, :], sample['all_masks'][:, :], *sample['image']]
        )
        if self.transform is not None:
            aug = self.transform(aug)
        
        return aug, sample['label']
