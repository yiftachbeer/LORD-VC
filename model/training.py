import itertools
from pathlib import Path
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import wandb

from model.lord import LatentModel, AmortizedModel, VGGDistance
from model.utils import AverageMeter, NamedTensorDataset


def train_latent(config, device, imgs, classes, model_dir: Path):
	latent_model = LatentModel(config)

	data = dict(
		img=torch.from_numpy(imgs),
		img_id=torch.arange(imgs.shape[0]).type(torch.int64),
		class_id=torch.from_numpy(classes.astype(np.int64))
	)

	dataset = NamedTensorDataset(data)
	data_loader = DataLoader(
		dataset, batch_size=config['train']['batch_size'],
		shuffle=True, sampler=None, batch_sampler=None,
		num_workers=1, pin_memory=True, drop_last=True
	)

	latent_model.init()
	latent_model.to(device)

	criterion = VGGDistance(config['perceptual_loss']['layers']).to(device)
	dvector = torch.jit.load('pretrained/dvector.pt', map_location=device)
	cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

	optimizer = Adam([
		{
			'params': latent_model.decoder.parameters(),
			'lr': config['train']['learning_rate']['generator']
		},
		{
			'params': itertools.chain(latent_model.content_embedding.parameters(), latent_model.class_embedding.parameters()),
			'lr': config['train']['learning_rate']['latent']
		}
	], betas=(0.5, 0.999))

	scheduler = CosineAnnealingLR(
		optimizer,
		T_max=config['train']['n_epochs'] * len(data_loader),
		eta_min=config['train']['learning_rate']['min']
	)

	visualized_imgs = []

	train_loss = AverageMeter()
	for epoch in range(config['train']['n_epochs']):
		latent_model.train()
		train_loss.reset()

		pbar = tqdm(iterable=data_loader)
		for batch in pbar:
			batch = {name: tensor.to(device) for name, tensor in batch.items()}

			optimizer.zero_grad(set_to_none=True)
			out = latent_model(batch['img_id'], batch['class_id'])

			content_penalty = torch.sum(out['content_code'] ** 2, dim=1).mean()
			dvector_orig = dvector(batch['img'].squeeze(1).transpose(1, 2))
			dvector_const = dvector(out['img'].squeeze(1).transpose(1, 2))
			speaker_loss = -cos_sim(dvector_orig, dvector_const).mean()
			loss = criterion(out['img'], batch['img']) + config['content_decay'] * content_penalty + speaker_loss

			loss.backward()
			optimizer.step()
			scheduler.step()

			train_loss.update(loss.item())
			pbar.set_description_str('epoch #{}'.format(epoch))
			pbar.set_postfix(loss=train_loss.avg)

		pbar.close()
		torch.save(latent_model.state_dict(), model_dir / 'latent.pth')
		wandb.save(str(model_dir / 'latent.pth'))

		with torch.no_grad():
			fixed_sample_img = generate_samples(latent_model, device, dataset, step=epoch)

		wandb.log({
			'loss': train_loss.avg,
			'decoder_lr': scheduler.get_last_lr()[0],
			'latent_lr': scheduler.get_last_lr()[1],
		}, step=epoch)

		wandb.log({f'generated-{epoch}': [wandb.Image(fixed_sample_img)]}, step=epoch)
		visualized_imgs.append(np.asarray(fixed_sample_img).transpose(2,0,1)[:3])

		if epoch % 5 == 0:
			wandb.log({f'video': [
				wandb.Video(np.array(visualized_imgs)),
			]}, step=epoch)


def train_amortized(config, device, latent_model, imgs, classes, model_dir: Path):
	amortized_model = AmortizedModel(config)
	amortized_model.decoder.load_state_dict(latent_model.decoder.state_dict())

	data = dict(
		img=torch.from_numpy(imgs),
		img_id=torch.arange(imgs.shape[0]).type(torch.int64),
		class_id=torch.from_numpy(classes.astype(np.int64))
	)

	dataset = NamedTensorDataset(data)
	data_loader = DataLoader(
		dataset, batch_size=config['train']['batch_size'],
		shuffle=True, sampler=None, batch_sampler=None,
		num_workers=1, pin_memory=True, drop_last=True
	)

	latent_model.to(device)
	amortized_model.to(device)

	reconstruction_criterion = VGGDistance(config['perceptual_loss']['layers']).to(device)
	embedding_criterion = nn.MSELoss()

	optimizer = Adam(
		params=amortized_model.parameters(),
		lr=config['train_encoders']['learning_rate']['max'],
		betas=(0.5, 0.999)
	)

	scheduler = CosineAnnealingLR(
		optimizer,
		T_max=config['train_encoders']['n_epochs'] * len(data_loader),
		eta_min=config['train_encoders']['learning_rate']['min']
	)

	visualized_imgs = []

	train_loss = AverageMeter()
	for epoch in range(config['train_encoders']['n_epochs']):
		latent_model.eval()
		amortized_model.train()

		train_loss.reset()

		pbar = tqdm(iterable=data_loader)
		for batch in pbar:
			batch = {name: tensor.to(device) for name, tensor in batch.items()}

			optimizer.zero_grad(set_to_none=True)

			target_content_code = latent_model.content_embedding(batch['img_id'])
			target_class_code = latent_model.class_embedding(batch['class_id'])

			out = amortized_model(batch['img'])

			loss_reconstruction = reconstruction_criterion(out['img'], batch['img'])
			loss_content = embedding_criterion(out['content_code'].reshape(target_content_code.shape), target_content_code)
			loss_class = embedding_criterion(out['class_code'], target_class_code)

			loss = loss_reconstruction + 10 * loss_content + 10 * loss_class

			loss.backward()
			optimizer.step()
			scheduler.step()

			train_loss.update(loss.item())
			pbar.set_description_str('epoch #{}'.format(epoch))
			pbar.set_postfix(loss=train_loss.avg)

		pbar.close()
		torch.save(amortized_model.state_dict(), model_dir / 'amortized.pth')
		wandb.save(str(model_dir / 'amortized.pth'))

		with torch.no_grad():
			fixed_sample_img = generate_samples_amortized(amortized_model, device, dataset, step=epoch)

		wandb.log({
			'loss-amortized': loss.item(),
			'rec-loss-amortized': loss_reconstruction.item(),
			'content-loss-amortized': loss_content.item(),
			'class-loss-amortized': loss_class.item(),

		}, step=epoch)

		wandb.log({f'generated-{epoch}': [wandb.Image(fixed_sample_img)]}, step=epoch)
		visualized_imgs.append(np.asarray(fixed_sample_img).transpose(2,0,1)[:3])

		if epoch % 5 == 0:
			wandb.log({f'video': [
				wandb.Video(np.array(visualized_imgs)),
			]}, step=epoch)


def generate_samples(latent_model, device, dataset, n_samples=4, step=None):
	latent_model.eval()

	img_idx = torch.from_numpy(np.random.RandomState(seed=1234).choice(len(dataset), size=n_samples, replace=False).astype(np.int64))

	samples = dataset[img_idx]
	samples = {name: tensor.to(device) for name, tensor in samples.items()}
	fig = plt.figure(figsize=(10, 10))
	if step:
		fig.suptitle(f'Step={step}')
	for i in range(n_samples):
		# Plot row headers (speaker)
		plt.subplot(n_samples + 1, n_samples + 1,
					n_samples + 1 + i * (n_samples + 1) + 1)
		plt.imshow(samples['img'][i, 0].detach().cpu().numpy(), cmap='inferno')
		plt.gca().invert_yaxis()
		plt.axis('off')

		# Plot column headers (content)
		plt.subplot(n_samples + 1, n_samples + 1, i + 2)
		plt.imshow(samples['img'][i, 0].detach().cpu().numpy(), cmap='inferno')
		plt.gca().invert_yaxis()
		plt.axis('off')

		for j in range(n_samples):
			plt.subplot(n_samples + 1, n_samples + 1,
						n_samples + 2 + i * (n_samples + 1) + j + 1)

			content_id = samples['img_id'][[j]]
			class_id = samples['class_id'][[i]]
			cvt = latent_model(content_id, class_id)['img'].squeeze().detach().cpu().numpy()

			if step % 5 == 0:
				np.savez(f'samples/{step}_{content_id.item()}({samples["class_id"][[j]].item()})to{class_id.item()}.npz', cvt)

			plt.imshow(cvt, cmap='inferno')
			plt.gca().invert_yaxis()
			plt.axis('off')

	buf = io.BytesIO()
	plt.savefig(buf, format='png')
	buf.seek(0)
	pil_img = Image.open(buf)
	return pil_img


def generate_samples_amortized(amortized_model, device, dataset, n_samples=4, step=None):
	amortized_model.eval()

	img_idx = torch.from_numpy(np.random.RandomState(seed=1234).choice(len(dataset), size=n_samples, replace=False).astype(np.int64))

	samples = dataset[img_idx]
	samples = {name: tensor.to(device) for name, tensor in samples.items()}
	fig = plt.figure(figsize=(10, 10))
	if step:
		fig.suptitle(f'Step={step}')
	for i in range(n_samples):
		# Plot row headers (speaker)
		plt.subplot(n_samples + 1, n_samples + 1,
					n_samples + 1 + i * (n_samples + 1) + 1)
		plt.imshow(samples['img'][i, 0].detach().cpu().numpy(), cmap='inferno')
		plt.gca().invert_yaxis()
		plt.axis('off')

		# Plot column headers (content)
		plt.subplot(n_samples + 1, n_samples + 1, i + 2)
		plt.imshow(samples['img'][i, 0].detach().cpu().numpy(), cmap='inferno')
		plt.gca().invert_yaxis()
		plt.axis('off')

		for j in range(n_samples):
			plt.subplot(n_samples + 1, n_samples + 1,
						n_samples + 2 + i * (n_samples + 1) + j + 1)

			content_img = samples['img'][[j]]
			class_img = samples['img'][[i]]
			cvt = amortized_model.convert(content_img, class_img)['img'].squeeze().detach().cpu().numpy()

			if step % 5 == 0:
				np.savez(f'samples/e{step}_{samples["img_id"][[j]].item()}({samples["class_id"][[j]].item()})to{samples["class_id"][[i]].item()}.npz', cvt)

			plt.imshow(cvt, cmap='inferno')
			plt.gca().invert_yaxis()
			plt.axis('off')

	buf = io.BytesIO()
	plt.savefig(buf, format='png')
	buf.seek(0)
	pil_img = Image.open(buf)
	return pil_img
