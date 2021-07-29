import itertools
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb

from model.loss import VGGDistance, SpeakerLoss


class AverageMeter:

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def train_latent(model, config, device, data_loader, callbacks):
	reconstruction_criterion = VGGDistance(config['perceptual_loss']['layers']).to(device)
	speaker_criterion = SpeakerLoss().to(device)

	optimizer = Adam([
		{
			'params': itertools.chain(model.content_embedding.parameters(), model.class_embedding.parameters()),
			'lr': config['train']['learning_rate']['latent']
		},
		{
			'params': model.decoder.parameters(),
			'lr': config['train']['learning_rate']['decoder']
		},
	], betas=(0.5, 0.999))

	scheduler = CosineAnnealingLR(
		optimizer,
		T_max=config['train']['n_epochs'] * len(data_loader),
		eta_min=config['train']['learning_rate']['min']
	)

	train_loss = AverageMeter()
	for epoch in range(config['train']['n_epochs']):
		model.train()
		train_loss.reset()

		pbar = tqdm(iterable=data_loader)
		for img_id, class_id, img in pbar:
			optimizer.zero_grad(set_to_none=True)
			out_img, out_content_code, out_class_code = model(img_id, class_id)

			content_penalty = torch.sum(out_content_code ** 2, dim=1).mean()
			reconstruction_loss = reconstruction_criterion(out_img, img)
			speaker_loss = speaker_criterion(out_img, img)
			loss = reconstruction_loss + config['content_decay'] * content_penalty + config['lambda_speaker'] * speaker_loss

			loss.backward()
			optimizer.step()
			scheduler.step()

			train_loss.update(loss.item())
			pbar.set_description_str('epoch #{}'.format(epoch))
			pbar.set_postfix(loss=train_loss.avg)

		pbar.close()

		for callback in callbacks:
			callback.on_epoch_end(model, epoch)

		wandb.log({
			'loss': loss.item(),
			'reconstruction-loss': reconstruction_loss.item(),
			'speaker-loss': speaker_loss.item(),
			'decoder-lr': scheduler.get_last_lr()[0],
			'latent-lr': scheduler.get_last_lr()[1],
		}, step=epoch)


def train_autoencoder(model, config, device, data_loader, callbacks):
	reconstruction_criterion = VGGDistance(config['perceptual_loss']['layers']).to(device)
	embedding_criterion = nn.MSELoss()

	optimizer = Adam(
		params=model.parameters(),
		lr=config['train_encoders']['learning_rate']['max'],
		betas=(0.5, 0.999)
	)

	scheduler = CosineAnnealingLR(
		optimizer,
		T_max=config['train_encoders']['n_epochs'] * len(data_loader),
		eta_min=config['train_encoders']['learning_rate']['min']
	)

	train_loss = AverageMeter()
	for epoch in range(config['train_encoders']['n_epochs']):
		model.train()

		train_loss.reset()

		pbar = tqdm(iterable=data_loader)
		for content_code, class_code, img in pbar:
			optimizer.zero_grad(set_to_none=True)

			out_img, out_content_code, out_class_code = model(img)

			loss_reconstruction = reconstruction_criterion(out_img, img)
			loss_content = embedding_criterion(out_content_code.reshape(content_code.shape), content_code)
			loss_class = embedding_criterion(out_class_code, class_code)

			loss = loss_reconstruction + 10 * loss_content + 10 * loss_class

			loss.backward()
			optimizer.step()
			scheduler.step()

			train_loss.update(loss.item())
			pbar.set_description_str('epoch #{}'.format(epoch))
			pbar.set_postfix(loss=train_loss.avg)

		pbar.close()

		for callback in callbacks:
			callback.on_epoch_end(model, epoch)

		wandb.log({
			'loss': loss.item(),
			'reconstruction-loss': loss_reconstruction.item(),
			'content-loss': loss_content.item(),
			'class-loss': loss_class.item(),
		}, step=epoch)

