from data.celldatamodule import CellDataModule
from data.celldata import CellDataset
from data.cellcrop import CellCrop
from model.model import CellSlighter
from data.utils import CellCropLoader
import lightning as L
import lightning.pytorch as pl
import wandb
from torchvision.transforms import v2
from lightning.pytorch.callbacks import ModelCheckpoint
from data.transforms import RandomSubsetTransform, poisson_sampling, cell_shape_augmentation, env_shape_augmentation
import torch
import os
from torch.utils.data import WeightedRandomSampler
from data.celldata import create_subset_dataset
from typing import List
import numpy as np
from icecream import ic


def create_weighted_random_sampler(labels: List[int], num_samples_per_class: int,
                                   num_classes: int) -> WeightedRandomSampler:
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    num_samples = num_samples_per_class * (num_classes - 1)
    return WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)


def main():
    wandb_logger = pl.loggers.WandbLogger(
        project='cellsighter2137'
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    train_dir = os.path.join(project_dir, 'train')
    cell_crop_loader = CellCropLoader(
        train_dir,
        cell_data_file='cell_data.h5ad',
        data_images_path='images_masks/img/',
        data_masks_path='images_masks/masks/',
        crop_size=120
    )
    train_transform = RandomSubsetTransform([
        poisson_sampling,
        cell_shape_augmentation,
        env_shape_augmentation,
    ])
    crops = cell_crop_loader.prepare_cell_crops()
    dataset = CellDataset(crops, train_transform)
    labels = [crop.label for crop in crops]
    num_classes = 15
    
    num_samples_per_class = min(torch.bincount(torch.tensor(labels), minlength=num_classes - 1)).item()
    # ic(num_samples_per_class)
    sampler = create_weighted_random_sampler(labels, num_samples_per_class, num_classes)
    # Create a subset of the dataset using the sampler
    subset_indices = list(iter(sampler))
    subset_dataset = create_subset_dataset(dataset, subset_indices)
    datamodule = CellDataModule(subset_dataset, batch_size=512, num_workers=0, pin_memory=True,
                                persistent_workers=False)
    del crops
    del dataset
    
    rng = torch.Generator()
    unique_id = torch.randint(0, 1000000, (1,), generator=rng).item()
    ckpt_filename = f'cellslighter-{unique_id}-{{epoch:02d}}-{{loss_val:.2f}}'
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename=ckpt_filename,
        save_top_k=3,
        monitor='loss_val',
        mode='min',
        save_last=True
    )
    trainer = L.Trainer(
        max_epochs=50,
        devices=1,
        accelerator='auto',
        precision="16",
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )
    model = CellSlighter(15, backbone='vit', img_size=120)
    trainer.fit(model, datamodule)
    wandb.finish()
    checkpoint_filenames = [os.path.basename(path) for path in checkpoint_callback.best_k_models.keys()]
    checkpoint_file = 'checkpoint_paths.txt'
    with open(checkpoint_file, 'w') as file:
        file.write('\n'.join(checkpoint_filenames))
    
    print(f'Checkpoint filenames saved to {checkpoint_file}')


if __name__ == '__main__':
    main()
