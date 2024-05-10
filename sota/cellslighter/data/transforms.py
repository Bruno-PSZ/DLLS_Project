import random
from typing import Sequence, Callable

import cv2
import numpy as np
import torch
from torchvision.transforms import v2

MASK_INDEX = 0
ENV_MASK_INDEX = 1


class RandomSubsetTransform(v2.Transform):
    def __init__(self, transforms: Sequence[Callable]):
        super().__init__()

        self.transforms = transforms

    def forward(self, x):
        transforms = random.sample(self.transforms, k=random.randint(0, len(self.transforms) - 1))

        for transform in transforms:
            x = transform(x)

        return x


def poisson_sampling(x):
    # Assume that the first two channels are masks
    blur = cv2.GaussianBlur(x[2:, :, :], (5, 5), 0)
    x[2:, :, :] = np.random.poisson(lam=blur, size=x[2:, :, :].shape)
    return x


def layer_shape_augmentation(x, layer_index):
    mask = x[layer_index, :, :]
    kernel_size = np.random.randint(3, 6)
    kernel = np.ones(kernel_size, np.uint8)
    img_dil = cv2.dilate(mask, kernel, iterations=1)
    x[layer_index, :, :] = img_dil
    return x


def cell_shape_augmentation(x):
    return layer_shape_augmentation(x, MASK_INDEX)


def env_shape_augmentation(x):
    return layer_shape_augmentation(x, ENV_MASK_INDEX)


train_transform = v2.Compose([
    RandomSubsetTransform([
        v2.Lambda(poisson_sampling),
        v2.Lambda(cell_shape_augmentation),
        v2.Lambda(env_shape_augmentation),
    ]),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    RandomSubsetTransform([
        v2.RandomRotation(degrees=(-180, 180)),
        v2.RandomVerticalFlip(p=0.75),
        v2.RandomHorizontalFlip(p=0.75)
    ]),
])
