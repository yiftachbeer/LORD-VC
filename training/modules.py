import itertools

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.loss import VGGDistance, SpeakerLoss


class LatentModule:

    def __init__(self, model, device, config, n_imgs):
        self.model = model
        self.reconstruction_criterion = VGGDistance(config['perceptual_loss']['layers']).to(device)
        self.speaker_criterion = SpeakerLoss().to(device)

        self.lambda_content = config['content_decay']
        self.lambda_speaker = config['lambda_speaker']

        self.optimizer = Adam([
            {
                'params': itertools.chain(model.content_embedding.parameters(), model.class_embedding.parameters()),
                'lr': config['train']['learning_rate']['latent']
            },
            {
                'params': model.decoder.parameters(),
                'lr': config['train']['learning_rate']['decoder']
            },
        ], betas=(0.5, 0.999))

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['train']['n_epochs'] * n_imgs,
            eta_min=config['train']['learning_rate']['min']
        )

    def train_step(self, batch):
        img_id, class_id, img = batch

        out_img, out_content_code, out_class_code = self.model(img_id, class_id)

        reconstruction_loss = self.reconstruction_criterion(out_img, img)
        content_penalty = torch.sum(out_content_code ** 2, dim=1).mean()
        speaker_loss = self.speaker_criterion(out_img, img)

        loss = reconstruction_loss + \
               self.lambda_content * content_penalty + \
               self.lambda_speaker * speaker_loss

        return {
            'loss': loss,
            'reconstruction-loss': reconstruction_loss,
            'content-penalty-loss': content_penalty,
            'speaker-loss': speaker_loss,
        }


class AutoEncoderModule:

    def __init__(self, model, device, config, n_imgs):
        self.model = model
        self.reconstruction_criterion = VGGDistance(config['perceptual_loss']['layers']).to(device)
        self.embedding_criterion = nn.MSELoss()

        self.optimizer = Adam(
            params=model.parameters(),
            lr=config['train_encoders']['learning_rate']['max'],
            betas=(0.5, 0.999)
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['train_encoders']['n_epochs'] * n_imgs,
            eta_min=config['train_encoders']['learning_rate']['min']
        )

    def train_step(self, batch):
        content_code, class_code, img = batch

        out_img, out_content_code, out_class_code = self.model(img)

        loss_reconstruction = self.reconstruction_criterion(out_img, img)
        loss_content = self.embedding_criterion(out_content_code.reshape(content_code.shape), content_code)
        loss_class = self.embedding_criterion(out_class_code, class_code)

        loss = loss_reconstruction + 10 * loss_content + 10 * loss_class

        return {
            'loss': loss,
            'reconstruction-loss': loss_reconstruction,
            'content-loss': loss_content,
            'class-loss': loss_class,
        }
