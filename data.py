import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset


def get_data(data_path):
    data = np.load(data_path)
    imgs = data['imgs']

    dataset = TensorDataset(
        torch.arange(imgs.shape[0]).type(torch.int64),
        torch.from_numpy(data['classes'].astype(np.int64)),
        torch.from_numpy(imgs)
    )

    return dataset, imgs, data


class DeviceDataLoader:
    # TODO is there not a builtin solution for this?

    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        for batch in self.dataloader:
            yield [tensor.to(self.device) for tensor in batch]


def get_dataloader(dataset, batch_size, device):
    return DeviceDataLoader(DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, sampler=None, batch_sampler=None,
        num_workers=1, pin_memory=True, drop_last=True
    ), device)


class LatentCodesDataLoader:

    def __init__(self, dataloader, latent_model):
        self.dataloader = dataloader
        self.latent_model = latent_model

    def __iter__(self):
        for batch in self.dataloader:
            img_id, class_id, img = batch
            content_code = self.latent_model.content_embedding(img_id)
            class_code = self.latent_model.class_embedding(class_id)

            yield content_code, class_code, img


def get_latent_codes_dataloader(dataset, latent_model, batch_size):
    return LatentCodesDataLoader(get_dataloader(dataset, batch_size), latent_model)
