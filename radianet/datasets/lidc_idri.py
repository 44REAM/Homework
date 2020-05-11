import pandas as pd
import numpy as np

from torch.utils.data import Dataset



class LIDCDataset(Dataset):

    def __init__(self, transforms, path):
        self.transforms = transforms
        self.path = path
        self.partition, self.labels = self.get_lidc_label()

    def get_lidc_label(self):
        # path to csv file
        path = self.path + 'df.csv'
        df = pd.read_csv(path)

        partition = df.nodule_name.to_numpy()
        label = df.nodule_label.to_numpy()

        return partition, label

    def get_data(self, idx):

        path = self.path + str(idx) + '.npy'
        data = np.load(path) 
        return data

    def __len__(self):
        return len(self.partition)

    def __getitem__(self, index):
        idx = self.partition[index]

        X = self.get_data(idx)

        X = self.transforms(X)

        y = self.labels[index]

        return X, y




if __name__ == "__main__":
    from ..config import Config
    from .transformers import Transforms
    from .utils import show

    transforms = Transforms()
    datasets = LIDCDataset(transforms, Config.LIDC_PATH)
    print(datasets.partition)
    print(datasets.labels)
    img, label = datasets[0]
    show(img)
