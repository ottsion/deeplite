from torchvision import datasets, transforms
from tqdm import tqdm

from base import BaseDataLoader
from data_loader.criteo_dataset import *
from data_loader.thucnews_dataset import ThucnewsDataset


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CriteoDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.data_dir = data_dir
        self.dataset = CriteoDataset(self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class ThucnewsDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.data_dir = data_dir
        self.dataset = ThucnewsDataset(self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)



