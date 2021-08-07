import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset


def load_data(data_path):
    data = np.load(data_path)
    imgs = data['imgs']

    dataset = TensorDataset(
        torch.arange(imgs.shape[0]).type(torch.int64),
        torch.from_numpy(data['classes'].astype(np.int64)),
        torch.from_numpy(imgs)
    )

    return dataset, imgs.shape[1:], imgs.shape[0], data['n_classes'].item()


class DeviceDataLoader:
    # TODO is there not a builtin solution for this?

    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for batch in self.dataloader:
            yield [tensor.to(self.device) for tensor in batch]


def get_common_dataloader(dataset, batch_size):
    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, sampler=None, batch_sampler=None,
        num_workers=1, pin_memory=True, drop_last=True
    )


def get_dataloader(dataset, batch_size, device):
    return DeviceDataLoader(get_common_dataloader(dataset, batch_size), device)


class LatentCodesDataLoader:

    def __init__(self, dataloader, content_embedding, class_embedding):
        self.dataloader = dataloader
        self.content_embedding = content_embedding
        self.class_embedding = class_embedding

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for batch in self.dataloader:
            img_id, class_id, img = batch
            content_code = self.content_embedding(img_id)
            class_code = self.class_embedding(class_id)

            yield content_code, class_code, img


def get_latent_codes_dataloader(dataset, batch_size, device, content_embedding, class_embedding):
    return DeviceDataLoader(
        LatentCodesDataLoader(
            get_common_dataloader(dataset, batch_size),
            content_embedding,
            class_embedding),
        device)
