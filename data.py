import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, TensorDataset


def load_data(data_path):
    data = np.load(data_path)
    imgs = data['imgs']

    dataset = TensorDataset(
        torch.arange(imgs.shape[0]).type(torch.int64),
        torch.from_numpy(data['classes'].astype(np.int64)),
        torch.from_numpy(imgs)
    )

    return dataset, imgs.shape[1:], imgs.shape[0], data['n_classes'].item()


class LatentCodesDataset(Dataset):

    def __init__(self, dataset, content_codes, class_codes):
        self.dataset = dataset
        self.content_codes = content_codes
        self.class_codes = class_codes

    def __getitem__(self, index):
        img_id, class_id, img = self.dataset[index]
        content_code = self.content_codes[img_id]
        class_code = self.class_codes[class_id]

        return content_code, class_code, img

    def __len__(self):
        return len(self.dataset)


class DeviceDataLoader:
    # TODO is there not a builtin solution for this? maybe collate_fn?

    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(tensor.to(self.device) for tensor in batch)


def get_dataloader(dataset, batch_size, device):
    return DeviceDataLoader(DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, sampler=None, batch_sampler=None,
        num_workers=1, pin_memory=True, drop_last=True
    ), device)
