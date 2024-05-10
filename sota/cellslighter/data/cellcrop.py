import numpy as np


class CellCrop:
    """Class representing a single cell crop. """
    
    def __init__(self, cell_id, img, mask, label, slices):
        self.cell_id = cell_id
        self.img = img
        self.mask = mask
        self.label = label
        self.slices = slices
    
    def sample(self):
        cropped_image = self.img[:, *self.slices]
        mask = self.mask[self.slices]
        
        return {
            'cell_id': self.cell_id,
            'image': cropped_image,
            'mask': (mask == self.cell_id).astype(np.float16),
            'all_masks': (mask > 0).astype(np.float16),
            'label': self.label
        }
