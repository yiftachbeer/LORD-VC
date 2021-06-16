import os
import itertools
import pickle
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

from model.modules import LatentModel, AmortizedModel, VGGDistance
from model.utils import AverageMeter, NamedTensorDataset


class Lord:

	def __init__(self, config=None):
		super().__init__()

		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.latent_model = None
		self.amortized_model = None

	def load(self, model_dir, latent=True, amortized=True):
		with open(os.path.join(model_dir, 'config.pkl'), 'rb') as config_fd:
			self.config = pickle.load(config_fd)

		if latent:
			self.latent_model = LatentModel(self.config)
			self.latent_model.load_state_dict(torch.load(os.path.join(model_dir, 'latent.pth')))

		if amortized:
			self.amortized_model = AmortizedModel(self.config)
			self.amortized_model.load_state_dict(torch.load(os.path.join(model_dir, 'amortized.pth')))

	def save(self, model_dir, latent=True, amortized=True):
		with open(os.path.join(model_dir, 'config.pkl'), 'wb') as config_fd:
			pickle.dump(self.config, config_fd)

		if latent:
			torch.save(self.latent_model.state_dict(), os.path.join(model_dir, 'latent.pth'))

		if amortized:
			torch.save(self.amortized_model.state_dict(), os.path.join(model_dir, 'amortized.pth'))

	def train_latent(self, imgs, classes, model_dir):
		self.latent_model = LatentModel(self.config)

		data = dict(
			img=torch.from_numpy(imgs),
			img_id=torch.arange(imgs.shape[0]).type(torch.int64),
			class_id=torch.from_numpy(classes.astype(np.int64))
		)

		dataset = NamedTensorDataset(data)
		data_loader = DataLoader(
			dataset, batch_size=self.config['train']['batch_size'],
			shuffle=True, sampler=None, batch_sampler=None,
			num_workers=1, pin_memory=True, drop_last=True
		)

		self.latent_model.init()
		self.latent_model.to(self.device)

		criterion = VGGDistance(self.config['perceptual_loss']['layers']).to(self.device)
		dvector = torch.jit.load('pretrained/dvector.pt', map_location=self.device)
		cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

		optimizer = Adam([
			{
				'params': self.latent_model.decoder.parameters(),
				'lr': self.config['train']['learning_rate']['generator']
			},
			{
				'params': itertools.chain(self.latent_model.content_embedding.parameters(), self.latent_model.class_embedding.parameters()),
				'lr': self.config['train']['learning_rate']['latent']
			}
		], betas=(0.5, 0.999))

		scheduler = CosineAnnealingLR(
			optimizer,
			T_max=self.config['train']['n_epochs'] * len(data_loader),
			eta_min=self.config['train']['learning_rate']['min']
		)

		visualized_imgs = []

		train_loss = AverageMeter()
		for epoch in range(self.config['train']['n_epochs']):
			self.latent_model.train()
			train_loss.reset()

			pbar = tqdm(iterable=data_loader)
			for batch in pbar:
				batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

				optimizer.zero_grad(set_to_none=True)
				out = self.latent_model(batch['img_id'], batch['class_id'])

				content_penalty = torch.sum(out['content_code'] ** 2, dim=1).mean()
				dvector_orig = dvector(batch['img'][:, 0, ...].transpose(1, 2))
				dvector_const = dvector(out['img'].transpose(1, 2))
				speaker_loss = -cos_sim(dvector_orig, dvector_const).mean()
				loss = criterion(out['img'][:, None, ...], batch['img']) + self.config['content_decay'] * content_penalty + speaker_loss

				loss.backward()
				optimizer.step()
				scheduler.step()

				train_loss.update(loss.item())
				pbar.set_description_str('epoch #{}'.format(epoch))
				pbar.set_postfix(loss=train_loss.avg)

			pbar.close()
			self.save(model_dir, latent=True, amortized=False)

			wandb.log({
				'loss': train_loss.avg,
				'decoder_lr': scheduler.get_last_lr()[0],
				'latent_lr': scheduler.get_last_lr()[1],
			}, step=epoch)

			with torch.no_grad():
				fixed_sample_img = self.generate_samples(dataset, step=epoch)

			wandb.log({f'generated-{epoch}': [wandb.Image(fixed_sample_img)]}, step=epoch)
			visualized_imgs.append(np.asarray(fixed_sample_img).transpose(2,0,1)[:3])

			if epoch % 5 == 0:
				wandb.log({f'video': [
					wandb.Video(np.array(visualized_imgs)),
				]}, step=epoch)

	def train_amortized(self, imgs, classes, model_dir):
		self.amortized_model = AmortizedModel(self.config)
		self.amortized_model.decoder.load_state_dict(self.latent_model.decoder.state_dict())

		data = dict(
			img=torch.from_numpy(imgs),
			img_id=torch.arange(imgs.shape[0]).type(torch.int64),
			class_id=torch.from_numpy(classes.astype(np.int64))
		)

		dataset = NamedTensorDataset(data)
		data_loader = DataLoader(
			dataset, batch_size=self.config['train']['batch_size'],
			shuffle=True, sampler=None, batch_sampler=None,
			num_workers=1, pin_memory=True, drop_last=True
		)

		self.latent_model.to(self.device)
		self.amortized_model.to(self.device)

		reconstruction_criterion = VGGDistance(self.config['perceptual_loss']['layers']).to(self.device)
		embedding_criterion = nn.MSELoss()

		optimizer = Adam(
			params=self.amortized_model.parameters(),
			lr=self.config['train_encoders']['learning_rate']['max'],
			betas=(0.5, 0.999)
		)

		scheduler = CosineAnnealingLR(
			optimizer,
			T_max=self.config['train_encoders']['n_epochs'] * len(data_loader),
			eta_min=self.config['train_encoders']['learning_rate']['min']
		)

		visualized_imgs = []

		train_loss = AverageMeter()
		for epoch in range(self.config['train_encoders']['n_epochs']):
			self.latent_model.eval()
			self.amortized_model.train()

			train_loss.reset()

			pbar = tqdm(iterable=data_loader)
			for batch in pbar:
				batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

				optimizer.zero_grad(set_to_none=True)

				target_content_code = self.latent_model.content_embedding(batch['img_id'])
				target_class_code = self.latent_model.class_embedding(batch['class_id'])

				out = self.amortized_model(batch['img'])

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
			self.save(model_dir, latent=False, amortized=True)

			wandb.log({
				'loss-amortized': loss.item(),
				'rec-loss-amortized': loss_reconstruction.item(),
				'content-loss-amortized': loss_content.item(),
				'class-loss-amortized': loss_class.item(),

			}, step=epoch)

			with torch.no_grad():
				fixed_sample_img = self.generate_samples_amortized(dataset, step=epoch)

			wandb.log({f'generated-{epoch}': [wandb.Image(fixed_sample_img)]}, step=epoch)
			visualized_imgs.append(np.asarray(fixed_sample_img).transpose(2,0,1)[:3])

			if epoch % 5 == 0:
				wandb.log({f'video': [
					wandb.Video(np.array(visualized_imgs)),
				]}, step=epoch)

	def generate_samples(self, dataset, n_samples=4, step=None):
		self.latent_model.eval()

		img_idx = torch.from_numpy(np.random.RandomState(seed=1234).choice(len(dataset), size=n_samples, replace=False).astype(np.int64))

		samples = dataset[img_idx]
		samples = {name: tensor.to(self.device) for name, tensor in samples.items()}
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
				cvt = self.latent_model(content_id, class_id)['img'][0].detach().cpu().numpy()

				if step % 5 == 0:
					np.savez(f'samples/{step}_{content_id.item()}({samples["class_id"][[j]].item()})to{class_id.item()}.npz', cvt)

				plt.imshow(cvt, cmap='inferno')
				plt.gca().invert_yaxis()
				plt.axis('off')

		buf = io.BytesIO()
		plt.savefig(buf, format='png')
		plt.show()
		buf.seek(0)
		pil_img = Image.open(buf)
		return pil_img

	def generate_samples_amortized(self, dataset, n_samples=4, step=None):
		self.amortized_model.eval()

		img_idx = torch.from_numpy(np.random.RandomState(seed=1234).choice(len(dataset), size=n_samples, replace=False).astype(np.int64))

		samples = dataset[img_idx]
		samples = {name: tensor.to(self.device) for name, tensor in samples.items()}
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
				cvt = self.amortized_model.convert(content_img, class_img)['img'][0].detach().cpu().numpy()

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
