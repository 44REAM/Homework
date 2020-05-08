import numpy as np
from .datahandler import Dataset

class SampleDataset3D():

    def __init__(self, n_sample = 100, channels = 1, width = 64, height = 64, deep = 12, train = True):
        self.data = np.random.randn(n_sample, channels, width, height, deep).astype(np.float32)
        self.labels = np.array([np.random.randint(0, 2) for x in range(n_sample) ]).astype(np.float32)
        self.list_ids = np.array([i for i in range(n_sample)])

    def get_data(self, idx):
        return self.data[idx]

    def get_dataset(self):
        train_dataset = Dataset(self.list_ids, self.labels, self)
        val_dataset = Dataset(self.list_ids, self.labels, self)
        dataset = {
            "train":train_dataset,
            "val":val_dataset
        }
        return dataset



if __name__ == '__main__':
    dataset = SampleDataset3D()
    dataset.get_dataset()

    