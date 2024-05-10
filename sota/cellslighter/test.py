from data.celldatamodule import CellDataModule
from data.celldata import CellDataset
from data.cellcrop import CellCrop
from model.model import CellSlighter
from data.utils import CellCropLoader
import lightning as L
import lightning.pytorch as pl
import wandb
from torchvision.transforms import v2
import torch
import os

pl.seed_everything(42, workers=True)

def main():
    wandb_logger = pl.loggers.WandbLogger(
        project='cellslighter2137-test'
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(script_dir, '..', 'test')
    checkpoint_dir = os.path.join(script_dir, 'checkpoints')
    checkpoint_file = 'checkpoint_paths.txt'
    with open(checkpoint_file, 'r') as file:
        checkpoint_filenames = file.read().splitlines()
    cell_crop_loader = CellCropLoader(
        test_dir,
        cell_data_file='cell_data.h5ad',
        data_images_path='images_masks/img/',
        data_masks_path='images_masks/masks/',
        crop_size=60
    )
    val_transform = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True)
    ])
    crops = cell_crop_loader.prepare_cell_crops()
    dataset = CellDataset(crops, val_transform)
    datamodule = CellDataModule(dataset, batch_size=1024)
    trainer = L.Trainer(
        devices=1,
        accelerator='gpu',
        precision="16-mixed",
        logger=wandb_logger
    )

    for checkpoint_filename in checkpoint_filenames:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
        model = CellSlighter.load_from_checkpoint(checkpoint_path)
        model.eval()
        model.freeze()
        trainer.validate(model, datamodule)
    
    wandb.finish()


if __name__ == '__main__':
    main()
