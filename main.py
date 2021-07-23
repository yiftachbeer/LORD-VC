import pickle
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import fire
import wandb
import soundfile as sf

import torch
import torchaudio

from training import train_latent, train_amortized
from config import base_config as config
from model.wav2mel import Wav2Mel
from model.lord import LatentModel, AmortizedModel
from callbacks import GenerateSamplesCallback


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
			for wav_file in sorted(spk.rglob('*mic2.flac')):
				mel = wav2mel(*torchaudio.load(wav_file))
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

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		with wandb.init(config=config):
			save_config(config, save_path)
			train_latent(
				config=config,
				device=device,
				imgs=imgs,
				classes=data['classes'],
				model_dir=Path(save_path),
				callback=GenerateSamplesCallback(device),
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

		latent_model = LatentModel(config)
		latent_model.load_state_dict(torch.load(Path(model_dir) / 'latent.pth'))

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		with wandb.init(config=config):
			save_config(config, model_dir)
			train_amortized(
				config=config,
				device=device,
				latent_model=latent_model,
				imgs=imgs,
				classes=data['classes'],
				model_dir=Path(model_dir),
				callback=GenerateSamplesCallback(device)
			)

	def convert(self, data_path, model_dir, content_file_path: str, speaker_file_path: str, output_path: str,
				vocoder_path: str = r"pretrained\vocoder.pth", **kwargs):
		data = np.load(data_path)
		imgs = data['imgs']

		config.update(dict(
			img_shape=imgs.shape[1:],
			n_imgs=imgs.shape[0],
			n_classes=data['n_classes'].item(),
		))

		update_nested(config, kwargs)

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		wav2mel = Wav2Mel()

		amortized_model = AmortizedModel(config)
		amortized_model.load_state_dict(torch.load(Path(model_dir) / 'amortized.pth'))
		amortized_model.to(device)
		amortized_model.eval()

		vocoder = torch.jit.load(vocoder_path, map_location=device)
		vocoder.eval()

		with torch.no_grad():
			content_mel = wav2mel(*torchaudio.load(content_file_path)).to(device)
			speaker_mel = wav2mel(*torchaudio.load(speaker_file_path)).to(device)

			converted_mel = amortized_model.convert(
				content_img=content_mel[None, None, ...],
				class_img=speaker_mel[None, None, ...]
			)['img']

			wav = vocoder.generate([converted_mel[0, 0].T])[0]
			sf.write(output_path, wav.data.cpu().numpy(), 16000)


if __name__ == '__main__':
	fire.Fire(Main())
