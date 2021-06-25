import pickle
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import fire
import wandb

import torch
import torchaudio

from model.training import Lord
from config import base_config as config
from model.wav2mel import Wav2Mel
from model.modules import LatentModel, AmortizedModel


def update_nested(d1: dict, d2: dict):
	"""
	Update d1, that might have nested dictionaries with the items of d2.
	Nested keys in d2 should be separated with a '/' character.
	Example:
	>> d1 = {'a': {'b': 1, 'c': 2}}
	>> d2 = {'a/b': 3, 'd': 4}
	>> update_nested(d1, d2)
	d1 = {'a': {'b': 3, 'c': 2}, 'd': 4}

	:param d1: The dict to update.
	:param d2: The values to update with.
	:return: The updated dict.
	"""
	for key, val in d2.items():
		key_path = key.split('/')
		curr_d = d1
		for key in key_path[:-1]:
			curr_d = curr_d[key]
		curr_d[key_path[-1]] = val

	return d1


def save_config(config, save_path):
	config_path = Path(save_path) / 'config.pkl'
	with open(config_path, 'wb') as config_fd:
		pickle.dump(config, config_fd)
	wandb.save(str(config_path))


class Main:

	def preprocess(self, data_dir: str, save_dest: str, segment: int = 128):
		wav2mel = Wav2Mel()

		cropped_mels = []
		classes = []
		file_names = []

		for i_spk, spk in enumerate(tqdm(sorted(Path(data_dir).glob('*')))):
			for wav_file in sorted((Path(data_dir) / spk).rglob('*mic2.flac')):
				speech_tensor, sample_rate = torchaudio.load(wav_file)
				mel = wav2mel(speech_tensor, sample_rate)
				if mel is not None and mel.shape[-1] > segment:
					start = mel.shape[-1] // 2 - segment // 2

					cropped_mels.append(mel[:, start:start + segment].numpy())
					classes.append(i_spk)
					file_names.append(str(wav_file))

		np.savez(file=save_dest,
				imgs=np.array(cropped_mels)[:, None, ...],  # add channel
				classes=np.array(classes),
				n_classes=np.unique(classes).size,
				file_names=file_names)

	def train(self, data_path: str, save_path: str, **kwargs):
		data = np.load(data_path)
		imgs = data['imgs']

		config.update(dict(
			img_shape=imgs.shape[1:],
			n_imgs=imgs.shape[0],
			n_classes=data['n_classes'].item(),
		))

		update_nested(config, kwargs)
		save_config(config, save_path)

		lord = Lord(config)
		with wandb.init(config=config):
			lord.train_latent(
				imgs=imgs,
				classes=data['classes'],
				model_dir=Path(save_path),
			)

	def train_encoders(self, data_path: str, model_dir: str, **kwargs):
		data = np.load(data_path)
		imgs = data['imgs']

		config.update(dict(
			img_shape=imgs.shape[1:],
			n_imgs=imgs.shape[0],
			n_classes=data['n_classes'].item(),
		))

		update_nested(config, kwargs)
		save_config(config, model_dir)

		lord = Lord(config)
		lord.latent_model = LatentModel(config)
		lord.latent_model.load_state_dict(torch.load(Path(model_dir) / 'latent.pth'))

		lord.load(Path(model_dir), latent=True, amortized=False)

		update_nested(lord.config, kwargs)

		with wandb.init(config=lord.config):
			lord.train_amortized(
				imgs=imgs,
				classes=data['classes'],
				model_dir=model_dir
			)

	def convert(self, model_dir):
		self.amortized_model = AmortizedModel(self.config)
		self.amortized_model.load_state_dict(torch.load(model_dir / 'amortized.pth'))

		# TODO
		raise NotImplemented


if __name__ == '__main__':
	fire.Fire(Main())
