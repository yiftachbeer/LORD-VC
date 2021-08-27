import itertools

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.loss import VGGDistance, SpeakerLoss


class LatentModule(nn.Module):

    def __init__(self, model, config, n_batches):
        super().__init__()

        self.lambda_content = config['content_decay']
        self.lambda_speaker = config['lambda_speaker']

        self.model = model
        self.reconstruction_criterion = VGGDistance(config['perceptual_loss']['layers'])
        if self.lambda_speaker != 0:
            self.speaker_criterion = SpeakerLoss()
        else:
            self.speaker_criterion = lambda x1, x2: 0

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
            T_max=config['train']['n_epochs'] * n_batches,
            eta_min=config['train']['learning_rate']['min']
        )

    def training_step(self, batch):
        img_id, class_id, img = batch

        out_img, out_content_code, out_class_code = self.model(img_id, class_id)

        reconstruction_loss = self.reconstruction_criterion(out_img.unsqueeze(1), img)
        content_penalty = self.lambda_content * torch.sum(out_content_code ** 2, dim=1).mean()
        speaker_loss = self.lambda_speaker * self.speaker_criterion(out_img.unsqueeze(1), img)

        loss = reconstruction_loss + content_penalty + speaker_loss

        return {
            'loss': loss,
            'reconstruction-loss': reconstruction_loss.detach(),
            'content-penalty-loss': content_penalty.detach(),
            'speaker-loss': speaker_loss.detach(),
        }


class AutoEncoderModule(nn.Module):

    def __init__(self, model, config, n_batches):
        super().__init__()

        self.model = model
        self.reconstruction_criterion = VGGDistance(config['perceptual_loss']['layers'])
        self.embedding_criterion = nn.MSELoss()

        self.optimizer = Adam(
            params=model.parameters(),
            lr=config['train_encoders']['learning_rate']['max'],
            betas=(0.5, 0.999)
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['train_encoders']['n_epochs'] * n_batches,
            eta_min=config['train_encoders']['learning_rate']['min']
        )

    def training_step(self, batch):
        content_code, class_code, img = batch

        out_img, out_content_code, out_class_code = self.model(img.squeeze(1))

        loss_reconstruction = self.reconstruction_criterion(out_img.unsqueeze(1), img)
        loss_content = 10 * self.embedding_criterion(out_content_code.reshape(content_code.shape), content_code)
        loss_class = 10 * self.embedding_criterion(out_class_code, class_code)

        loss = loss_reconstruction + loss_content + loss_class

        return {
            'loss': loss,
            'reconstruction-loss': loss_reconstruction.detach(),
            'content-loss': loss_content.detach(),
            'class-loss': loss_class.detach(),
        }
