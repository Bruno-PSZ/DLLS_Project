import random
from typing import Sequence, Callable, Tuple, List

import cv2
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import v2
from icecream import ic

MASK_INDEX = 0
ENV_MASK_INDEX = 1
TensorList = List[torch.Tensor]

from typing import Sequence, Callable, Tuple
import random
from torchvision.transforms import v2


class RandomSubsetTransform(v2.Transform):
    def __init__(self, transforms: Sequence[Callable[[TensorList | Tuple[TensorList, int]], TensorList]]):
        super().__init__()
        self.transforms = transforms

    def forward(self, x: TensorList) -> TensorList:
        #ic('RandomSubsetTransform_forward_26', x)
        transforms = random.sample(self.transforms, k=random.randint(0, len(self.transforms) - 1))
        #ic('RandomSubsetTransform_forward_28', transforms)
        for transform in transforms:
            x = transform(x, 1)

        return x


def poisson_sampling(x: TensorList, dummy: int) -> TensorList:
    # Apply poisson sampling to the image tensor
    #ic('poisson_sampling_37', x)
    blur = v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))(x[2])
    x[2] = torch.poisson(blur).to(torch.float16)
    return x


def layer_shape_augmentation(x: TensorList, layer_index: int) -> TensorList:
    #ic('layer_shape_augmentation_44', x)
    mask: np.ndarray = x[layer_index].numpy().astype(np.uint8)
    kernel_size: int = np.random.randint(3, 6)
    kernel: np.ndarray = np.ones((kernel_size, kernel_size), np.uint8)
    img_dil: np.ndarray = cv2.dilate(mask, kernel, iterations=1)
    x[layer_index] = torch.from_numpy(img_dil).to(torch.float32)
    return x


def cell_shape_augmentation(x: TensorList, dummy: int) -> TensorList:
    #ic('cell_shape_augmentation_54', x)
    return layer_shape_augmentation(x, MASK_INDEX)


def env_shape_augmentation(x: TensorList, dummy: int) -> TensorList:
    #ic('env_shape_augmentation_59', x)
    return layer_shape_augmentation(x, ENV_MASK_INDEX)


'''train_transform = v2.Compose([
    RandomSubsetTransform([
        v2.Lambda(poisson_sampling),
        v2.Lambda(cell_shape_augmentation),
        v2.Lambda(env_shape_augmentation),
        ]),
    ]),
])'''
