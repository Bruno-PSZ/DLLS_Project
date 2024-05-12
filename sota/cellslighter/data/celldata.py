from typing import List

import numpy as np
from torchvision.datasets import VisionDataset
from torch.utils.data import Subset
from .cellcrop import CellCrop
import torch

from typing import List, Tuple, Any
import torch
from torchvision.datasets import VisionDataset


class CellDataset(VisionDataset):
    def __init__(self, crops: List[CellCrop], transform: Any = None) -> None:
        super().__init__()
        self.crops = crops
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.crops)
    
    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample: dict[str, Any] = self.crops[item].sample()
        mask: torch.Tensor = torch.from_numpy(sample['mask'][:, :]).unsqueeze(0).to(torch.float32)
        all_mask: torch.Tensor = torch.from_numpy(sample['all_masks'][:, :]).unsqueeze(0).to(torch.float32)
        image: torch.Tensor = torch.from_numpy(sample['image']).to(torch.float32)
        label: torch.Tensor = torch.tensor(sample['label'], dtype=torch.uint8)
        x: List[torch.Tensor] = [mask, all_mask, image]
        if self.transform is not None:
            [mask, all_mask, image] = self.transform(x)  # TODO: check if types are correct for v2
            # transforms
        return (mask, all_mask, image, label)  # TODO: remember that the labels are strings and we need to
        # use encoder to convert them to tensors!


def create_subset_dataset(dataset: CellDataset, indices: List[int]) -> Subset:
    return Subset(dataset, indices)
