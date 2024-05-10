import lightning as L
from torch.utils.data import DataLoader, random_split

from data.celldata import CellDataset
from icecream import ic

class CellDataModule(L.LightningDataModule):
    def __init__(self, dataset: CellDataset, batch_size: int = 16):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self, stage=None):
        n = len(self.dataset)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        n_test = n - n_train - n_val
        self.train, self.val, self.test = random_split(self.dataset, [n_train, n_val, n_test])
        #ic(n)
        ##ic(len(self.train))
        #ic(len(self.val))
        #ic(len(self.test))
        #ic(self.train, self.val, self.test)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True, persistent_workers=True) #TODO

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, drop_last=True, num_workers=2, pin_memory=True, persistent_workers=True) #TODO

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, drop_last=True, num_workers=2, pin_memory=True, persistent_workers=True) #TODO
