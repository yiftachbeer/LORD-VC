import os
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import fire
import wandb

import torchaudio

from model.training import Lord
from config import base_config
from model.wav2mel import Wav2Mel


class Main:

	def preprocess(self, data_dir: str, save_dest: str, segment: int = 128):
		wav2mel = Wav2Mel()

		cropped_mels = []
		classes = []
		file_names = []

		for i_spk, spk in enumerate(tqdm(sorted(os.listdir(data_dir)))):
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
