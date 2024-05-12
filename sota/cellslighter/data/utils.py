import dataclasses
import os
from functools import cache
from typing import Dict
from icecream import ic
import numpy as np
import pyometiff
import scanpy as sc
import scipy.ndimage as ndimage

from .cellcrop import CellCrop

CELL_DATA_FILE = "cell_data.h5ad"
DATA_IMAGES_PATH = "images_masks/img/"
DATA_MASKS_PATH = "images_masks/masks/"
CELL_TYPES = ['plasma', 'B', 'CD4', 'Treg', 'BnT', 'Tumor', 'NK', 'HLADR', 'Neutrophil', 'CD8', 'pDC', 'DC', 'MacCD163',
              'Mural']
CELL_TYPE_DICT = {ctype: cid for ctype, cid in enumerate(CELL_TYPES)} | {cid: ctype for ctype, cid in
                                                                            enumerate(CELL_TYPES)}


@dataclasses.dataclass(eq=True, frozen=True)
class CellId:
    cell_number: int
    image_name: str


class CellCropLoader:
    def __init__(self,
                 root_dir: str,
                 cell_data_file: str = CELL_DATA_FILE,
                 data_images_path: str = DATA_IMAGES_PATH,
                 data_masks_path: str = DATA_MASKS_PATH,
                 crop_size=120):
        self.root_dir = root_dir
        self.cell_data_file = cell_data_file
        self.data_images_path = data_images_path
        self.data_masks_path = data_masks_path
        self.crop_size = crop_size

    @cache
    def _load_image(self, image_name: str):
        print("Loading image", image_name)
        image = self._read_tiff(os.path.join(self.root_dir, self.data_images_path, image_name))
        mask = self._read_tiff(os.path.join(self.root_dir, self.data_masks_path, image_name))

        # NOTE: Please use an arcsinh(x / 5.) transform on your data.
        image = np.pad(
            np.arcsinh(image / 5.), ((0, 0), (self.crop_size // 2, self.crop_size // 2),
            (self.crop_size // 2, self.crop_size // 2)))
        mask = np.pad(mask, ((self.crop_size // 2, self.crop_size // 2), (self.crop_size // 2, self.crop_size // 2)))
      #  ic("CellCropLoader", image.shape)
      #  ic("CellCropLoader", mask.shape)
        return image, mask

    @classmethod
    def _read_tiff(cls, fpath):
        image_reader = pyometiff.OMETIFFReader(fpath=fpath)
        image_array, _, _ = image_reader.read()
        return image_array

    def _get_cell_to_label_mapping(self) -> Dict[CellId, int]:
        cell_data = sc.read_h5ad(os.path.join(self.root_dir, self.cell_data_file))

        mapping = {}

        for i in range(cell_data.n_obs):
            cell = cell_data.obs.iloc[i]
            cell_id = CellId(cell_number=cell['ObjectNumber'], image_name=cell['image'])
            mapping[cell_id] = cell['cell_labels']

        return mapping

    def _extend_slices_to_crop_size(self, slices, bounds):
        new_slices = []
        for slc, max_size in zip(slices, bounds):
            start, stop = slc.start, slc.stop
            diff = self.crop_size - (stop - start)
            start = max(0, start - diff // 2)
            stop = min(max_size, stop + (diff + 1) // 2)
            new_slices.append(slice(start, stop))
            # else:
                # raise ValueError(f"Slice {slc} is larger than crop size {self.crop_size}")

        return tuple(new_slices)

    def _get_image_mask_and_slices(self, cell_id: CellId, slices: Dict[CellId, slice] = {}):
        image, mask = self._load_image(cell_id.image_name)

        if cell_id not in slices:
            objs = ndimage.find_objects(mask)

            for cell_number, obj in enumerate(objs, 1):
                if obj is not None:
                    slices[CellId(cell_number, cell_id.image_name)] = obj

        assert cell_id in slices

        return image, mask, slices.pop(cell_id)

    def prepare_cell_crops(self):
        cell_data = sc.read_h5ad(os.path.join(self.root_dir, self.cell_data_file))
        crops = []
        num_classes = len(CELL_TYPE_DICT)
        class_other = 14

        for i in range(cell_data.n_obs):
            if i % 10000 == 0:
                print(f"Processed {i}/{cell_data.n_obs} cells")
            #if i > 30000:
            #   print("Stopping at 10000 cells")
            #   break
            cell = cell_data.obs.iloc[i]
            cell_id = CellId(cell_number=cell['ObjectNumber'], image_name=cell['image'])

            image, mask, slices = self._get_image_mask_and_slices(cell_id)

            slices = self._extend_slices_to_crop_size(slices, mask.shape)

            flag = 0
            for slc in slices:
                if slc.stop - slc.start != self.crop_size:
                    flag = 1
            if flag == 0:
                label = CELL_TYPE_DICT.get(cell['cell_labels'], class_other)
                crops.append(CellCrop(cell_id.cell_number, image, mask, label, slices))

        return crops
