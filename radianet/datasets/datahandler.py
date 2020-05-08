import torch
from torch.utils import data



class Dataset(data.Dataset):

  def __init__(self, list_ids, labels, dataset):

        self.labels = labels
        self.list_ids = list_ids
        self.dataset = dataset

  def __len__(self):
        return len(self.list_ids)

  def __getitem__(self, index):
        idx = self.list_ids[index]

        X = self.dataset.get_data(idx)
        y = self.labels[index]

        return X, y
