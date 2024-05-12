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
        crop_size=90
    )
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
    crops = cell_crop_loader.prepare_cell_crops()
    dataset = CellDataset(crops, train_transform)
    datamodule = CellDataModule(dataset, batch_size=128, num_workers=1, pin_memory=True, persistent_workers=True)

    rng = torch.Generator()
    unique_id = torch.randint(0, 1000000, (1,), generator=rng).item()
    ckpt_filename = f'cellslighter-{unique_id}-{{epoch:02d}}-{{loss_val:.2f}}'
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename=ckpt_filename,
        save_top_k=3,
        monitor='loss_val',
        mode='min',
        save_last=True,
    )
    trainer = L.Trainer(
        max_epochs=2,
        devices=1,
        accelerator='auto',
        precision="16",
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )
    model = CellSlighter(15)
    trainer.fit(model, datamodule)
    wandb.finish()
    checkpoint_filenames = [os.path.basename(path) for path in checkpoint_callback.best_k_models.keys()]
    checkpoint_file = 'checkpoint_paths.txt'
    with open(checkpoint_file, 'w') as file:
        file.write('\n'.join(checkpoint_filenames))

    print(f'Checkpoint filenames saved to {checkpoint_file}')
    
    
    
if __name__ == '__main__':
    main()
