import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class NamedTensorDataset(Dataset):

    def __init__(self, named_tensors):
        assert all(list(named_tensors.values())[0].size(0) == tensor.size(0) for tensor in named_tensors.values())
        self.named_tensors = named_tensors

    def __getitem__(self, index):
        return {name: tensor[index] for name, tensor in self.named_tensors.items()}

    def __len__(self):
        return list(self.named_tensors.values())[0].size(0)

    def subset(self, indices):
        return NamedTensorDataset(self[indices])


def get_data(data_path, batch_size):
    data = np.load(data_path)
    imgs = data['imgs']

    dataset = NamedTensorDataset(dict(
        img=torch.from_numpy(imgs),
        img_id=torch.arange(imgs.shape[0]).type(torch.int64),
        class_id=torch.from_numpy(data['classes'].astype(np.int64))
    ))

    data_loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, sampler=None, batch_sampler=None,
        num_workers=1, pin_memory=True, drop_last=True
    )

    return dataset, data_loader, imgs, data
