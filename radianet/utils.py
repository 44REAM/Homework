
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler


def split_dataset():
    pass


def split_dataloader(dataset, batch_size, validation_split=0.2,
                     shuffle_dataset=True, random_seed=42):

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler, shuffle=False)

    return train_loader, validation_loader


def get_dataloader(dataset, batch_size, validation_split=0.2,
                   shuffle_dataset=True, random_seed=42,
                   method='split_dataloader'):
    if method == 'split_dataloader':
        train_loader, validation_loader = split_dataloader(
            dataset, batch_size,
            validation_split=validation_split,
            shuffle_dataset=shuffle_dataset, random_seed=random_seed
        )
        dataloader = {
            'train': train_loader,
            'val': validation_loader
        }
        return dataloader
    return None
