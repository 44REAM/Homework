# pylint: disable=redefined-outer-name

import numpy as np
from torch.utils.data import Dataset


class SampleDataset3D(Dataset):

    def __init__(self, transforms, n_sample=100, channels=1, width=64, height=64, deep=12):
        np.random.seed(52)
        self.data = np.random.randn(
            n_sample, channels, width, height, deep).astype(np.float32)
        self.labels = np.array([np.random.randint(0, 2)
                                for x in range(n_sample)]).astype(np.float32)
        self.list_ids = np.array([i for i in range(n_sample)])
        self.transforms = transforms

    def get_data(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        idx = self.list_ids[index]

        X = self.get_data(idx)
        X = self.transforms(X)
        y = self.labels[index]

        return X, y


class SampleDataset2D(Dataset):

    def __init__(self, transforms, n_sample=100, channels=3, width=224, height=244):
        np.random.seed(52)
        self.data = np.random.randn(
            n_sample, width, height, channels).astype(np.float32)
        self.labels = np.array([np.random.randint(0, 2)
                                for x in range(n_sample)]).astype(np.float32)
        self.list_ids = np.array([i for i in range(n_sample)])
        self.transforms = transforms

    def get_data(self, idx):

        return self.data[idx]

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        idx = self.list_ids[index]

        X = self.get_data(idx)
        X = self.transforms(X)
        y = self.labels[index]

        return X, y


if __name__ == '__main__':
    from ..utils import get_dataloader
    from ..config import Config
    from .transformers import Transforms

    config = Config
    transformers = Transforms()
    dataset = SampleDataset2D(transformers, n_sample=10)
    dataloader = get_dataloader(dataset, config.BATCHSIZE)

    for x, y in dataloader['train']:
        print(y)

    for x, y in dataloader['val']:
        print(y)
