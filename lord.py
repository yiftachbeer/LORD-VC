import numpy as np
import fire
import wandb

from model.training import Lord
from config import base_config


class Main:

	def train(self, data_path: str, save_path: str):
		data = np.load(data_path)
		imgs = data['imgs']

		config = dict(
			img_shape=imgs.shape[1:],
			n_imgs=imgs.shape[0],
			n_classes=data['n_classes'].item(),
		)

		config.update(base_config)

		lord = Lord(config)

		with wandb.init(config=config):
			lord.train_latent(
				imgs=imgs,
				classes=data['classes'],
				model_dir=save_path,
			)

	def train_encoders(self, data_path: str, model_dir: str):
		data = np.load(data_path)
		imgs = data['imgs']

		lord = Lord()
		lord.load(model_dir, latent=True, amortized=False)

		with wandb.init(config=lord.config):
			lord.train_amortized(
				imgs=imgs,
				classes=data['classes'],
				model_dir=model_dir
			)


if __name__ == '__main__':
	fire.Fire(Main())
