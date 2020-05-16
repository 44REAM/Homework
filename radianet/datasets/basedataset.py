from torch.utils.data import Dataset
import numpy as np

class BaseDataset(Dataset):
    def __init__(self, dataset, transform = True):
        self.dataset = dataset
        self.transform = transform
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X, y = self.dataset[index]
        if self.transform == True:
            if self.dataset.transforms != None:
                X = self.dataset.transforms(X)
            else:
                pass

        X = np.transpose(X, (2, 0, 1)).astype(np.float32)
        return X,y