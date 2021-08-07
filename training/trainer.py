from tqdm import tqdm
import wandb


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


class Trainer:

	def __init__(self, device):
		self.device = device

	def fit(self, module, data_loader, n_epochs, callbacks):
		module.to(self.device)
		model = module.model

		train_loss = AverageMeter()
		for epoch in range(n_epochs):
			model.train()
			train_loss.reset()

			pbar = tqdm(iterable=data_loader)
			for batch in pbar:
				batch = tuple(tensor.to(self.device) for tensor in batch)

				module.optimizer.zero_grad(set_to_none=True)

				train_result = module.train_step(batch)
				loss = train_result['loss']
				loss.backward()

				module.optimizer.step()
				module.scheduler.step()

				train_loss.update(loss.item())
				pbar.set_description_str('epoch #{}'.format(epoch))
				pbar.set_postfix(loss=train_loss.avg)

			pbar.close()

			for callback in callbacks:
				callback.on_epoch_end(model, epoch)

			wandb.log(train_result, step=epoch)
