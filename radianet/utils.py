
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler


def split_dataset():
    pass


def split_dataloader(dataset, batch_size, validation_split=0.2,
                     shuffle_dataset=True, random_seed=42, test_set = 'same'):

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
    
    if test_set == 'same':
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=len(valid_sampler),
                                                    sampler=valid_sampler, shuffle=False)
    else:
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=len(valid_sampler),
                                                    sampler=valid_sampler, shuffle=False)

    return train_loader, validation_loader, test_loader


def get_dataloader(dataset, batch_size, validation_split=0.2,
                   shuffle_dataset=True, random_seed=42,
                   method='split_dataloader', test_set = 'same'):
    if method == 'split_dataloader':
        train_loader, validation_loader, test_loader = split_dataloader(
            dataset, batch_size,
            validation_split=validation_split,
            shuffle_dataset=shuffle_dataset, random_seed=random_seed,
            test_set = test_set
        )

        dataloader = {
            'train': train_loader,
            'val': validation_loader,
            'test': test_loader
        }
        return dataloader
    return None
