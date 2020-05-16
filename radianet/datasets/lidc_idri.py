import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from .utils import get_dataloader


class LIDCDataset(Dataset):

    def __init__(self, transforms, path):
        self.transforms = transforms
        self.path = path
        self.partition, self.labels = self.get_lidc_label()
        self.labels = self.labels.astype(np.float32)

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

    def count_labels(self):
        neg = 0
        pos = 0
        for label in self.labels:
            if label == 0:
                neg += 1
            elif label == 1:
                pos += 1
        print("benign", neg)
        print("malignant", pos)

    def __len__(self):
        return len(self.partition)

    def __getitem__(self, index):
        idx = self.partition[index]

        X = self.get_data(idx)
        #X = self.transforms(X)
        y = self.labels[index]

        return X, y


def lidc_dataloader(path, config, validation_split=0.2, transforms=None,
                 shuffle_dataset=True, random_seed=42):

    dataset = LIDCDataset(transforms, path)
    dataloader = get_dataloader(dataset, config.BATCHSIZE,
                                shuffle_dataset=shuffle_dataset, random_seed=random_seed)
    return dataloader

def count_lidc(dataset):
    zero = 0
    one = 0
    for i in range(len(dataset)):
        _, y = dataset[i]
        if y == 0:
            zero+=1
        else:
            one+=1
    print(zero)
    print(one)

def count_testloader(testloader):
    for inputs, targets in testloader:

        print('zero',len(targets[targets==0]))
        print('zero',len(targets[targets==1]))
        


if __name__ == "__main__":
    from ..config import Config
    from .transformers import Transforms
    from .utils import show, print_dataloader

    transforms = Transforms()
    dataloader = lidc_dataloader(Config.LIDC_PATH, Config, transforms = transforms)


    print('train')
    print_dataloader(dataloader['train'])
    print('val')
    print_dataloader(dataloader['val'])
    print('test')
    print_dataloader(dataloader['test'])

    dataset = LIDCDataset(transforms = transforms, path = Config.LIDC_PATH)
    count_lidc(dataset)
    count_testloader(dataloader['test'])

