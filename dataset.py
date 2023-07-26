import os
from torchvision.datasets import EMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl

# Lightning way: for custom and predefined datasets
class LightningDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def prepare_data(self):
        EMNIST(self.data_dir, train=True, split='balanced', download=True)
        EMNIST(self.data_dir, train=False, split='balanced', download=True)

    def setup(self, stage):
        dataset = EMNIST(self.data_dir, 
                        train=True, 
                        split='balanced', 
                        download=False,
                        transform=ToTensor())

        self.train_set, self.val_set = random_split(dataset, [round(0.9*len(dataset)), round(0.1*len(dataset))])
        self.test_set = EMNIST(self.data_dir, 
                        train=False, 
                        split='balanced', 
                        download=False,
                        transform=ToTensor())

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

# Even faster way using a the predefined EMNIST from PyTorch
# dataset = EMNIST(os.getcwd(), split='balanced', download=True, transform=ToTensor())
# train_loader = DataLoader(dataset, num_workers=4, batch_size=200)
