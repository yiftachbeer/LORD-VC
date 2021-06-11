import numpy as np
import fire
import wandb

import dataset
from assets import AssetManager
from model.training import Lord
from config import base_config


class Main:

	def preprocess(self, base_dir: str, dataset_id: str, data_name: str, dataset_path: str = None):
		assets = AssetManager(base_dir)

		img_dataset = dataset.get_dataset(dataset_id, dataset_path)
		imgs, classes, contents = img_dataset.read_images()
		n_classes = np.unique(classes).size

		np.savez(
			file=assets.get_preprocess_file_path(data_name),
			imgs=imgs, classes=classes, contents=contents, n_classes=n_classes
		)

	def split_classes(self, base_dir: str, input_data_name: str, train_data_name: str, test_data_name: str,
					  num_test_classes: int):
		assets = AssetManager(base_dir)

		data = np.load(assets.get_preprocess_file_path(input_data_name))
		imgs, classes, contents = data['imgs'], data['classes'], data['contents']

		n_classes = np.unique(classes).size
		test_classes = np.random.choice(n_classes, size=num_test_classes, replace=False)

		test_idx = np.isin(classes, test_classes)
		train_idx = ~np.isin(classes, test_classes)

		np.savez(
			file=assets.get_preprocess_file_path(test_data_name),
			imgs=imgs[test_idx], classes=classes[test_idx], contents=contents[test_idx], n_classes=n_classes
		)

		np.savez(
			file=assets.get_preprocess_file_path(train_data_name),
			imgs=imgs[train_idx], classes=classes[train_idx], contents=contents[train_idx], n_classes=n_classes
		)

	def split_samples(self, base_dir: str, input_data_name: str, train_data_name: str, test_data_name: str,
					  test_split: float):
		assets = AssetManager(base_dir)

		data = np.load(assets.get_preprocess_file_path(input_data_name))
		imgs, classes, contents = data['imgs'], data['classes'], data['contents']

		n_classes = np.unique(classes).size
		n_samples = imgs.shape[0]

		n_test_samples = int(n_samples * test_split)

		test_idx = np.random.choice(n_samples, size=n_test_samples, replace=False)
		train_idx = ~np.isin(np.arange(n_samples), test_idx)

		np.savez(
			file=assets.get_preprocess_file_path(test_data_name),
			imgs=imgs[test_idx], classes=classes[test_idx], contents=contents[test_idx], n_classes=n_classes
		)

		np.savez(
			file=assets.get_preprocess_file_path(train_data_name),
			imgs=imgs[train_idx], classes=classes[train_idx], contents=contents[train_idx], n_classes=n_classes
		)

	def train(self, base_dir: str, data_name: str, model_name: str):
		assets = AssetManager(base_dir)
		model_dir = assets.recreate_model_dir(model_name)

		data = np.load(assets.get_preprocess_file_path(data_name))
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
				model_dir=model_dir,
			)

		lord.save(model_dir, latent=True, amortized=False)

	def train_encoders(self, base_dir: str, data_name: str, model_name: str):
		assets = AssetManager(base_dir)
		model_dir = assets.get_model_dir(model_name)

		data = np.load(assets.get_preprocess_file_path(data_name))
		imgs = data['imgs']

		lord = Lord()
		lord.load(model_dir, latent=True, amortized=False)

		with wandb.init(config=lord.config):
			lord.train_amortized(
				imgs=imgs,
				classes=data['classes'],
				model_dir=model_dir
			)

		lord.save(model_dir, latent=False, amortized=True)


if __name__ == '__main__':
	fire.Fire(Main())
