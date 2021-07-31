from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import fire
import wandb

import torch

from data import load_data, get_dataloader, get_latent_codes_dataloader
from training import train_latent, train_autoencoder
from config import get_config, save_config
from model.wav2mel import Wav2Mel, Mel2Wav
from model.adain_vc import get_latent_model, get_autoencoder
from model.lord import AutoEncoder
from callbacks import PlotTransferCallback, GenerateAudioSamplesCallback, GenerateEvaluationAudioSamplesCallback, \
	SaveCheckpointCallback, SaveModelCallback


class Main:

	def preprocess(self, data_dir: str, save_dest: str, segment: int = 128):
		wav2mel = Wav2Mel()

		cropped_mels = []
		classes = []
		file_names = []

		for i_spk, spk in enumerate(tqdm(sorted(Path(data_dir).glob('*')))):
			for wav_file in sorted(spk.rglob('*mic2.flac')):
				mel = wav2mel.parse_file(wav_file)
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
		dataset, img_shape, n_imgs, n_classes = load_data(data_path)
		config = get_config(img_shape, n_imgs, n_classes, kwargs=kwargs)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		model = get_latent_model(config)
		model.init()
		model.to(device)

		data_loader = get_dataloader(dataset, config['train']['batch_size'], device)

		with wandb.init(job_type='latent', config=config):
			save_config(config, Path(save_path) / 'config.pkl')
			train_latent(
				model=model,
				config=config,
				device=device,
				data_loader=data_loader,
				callbacks=[
					PlotTransferCallback(dataset, device, is_latent=True),
					GenerateAudioSamplesCallback(dataset, Path('samples_latent'), device),
					SaveCheckpointCallback(Path(save_path) / 'latent.ckpt')],
			)

	def train_encoders(self, data_path: str, model_dir: str, **kwargs):
		dataset, img_shape, n_imgs, n_classes = load_data(data_path)
		config = get_config(img_shape, n_imgs, n_classes, kwargs=kwargs)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		latent_model = get_latent_model(config)
		latent_model.load_state_dict(torch.load(Path(model_dir) / 'latent.ckpt'))
		latent_model.to(device)
		latent_model.eval()

		autoencoder = get_autoencoder(config)
		autoencoder.to(device)
		autoencoder.decoder.load_state_dict(latent_model.decoder.state_dict())

		data_loader = get_latent_codes_dataloader(dataset, config['train_encoders']['batch_size'], device, latent_model)

		with wandb.init(job_type='encoders', config=config):
			save_config(config, Path(model_dir) / 'config.pkl')
			train_autoencoder(
				model=autoencoder,
				config=config,
				device=device,
				data_loader=data_loader,
				callbacks=[
					PlotTransferCallback(dataset, device, is_latent=False),
					GenerateEvaluationAudioSamplesCallback(dataset, Path('samples_encoder'), device),
					SaveCheckpointCallback(Path(model_dir) / 'autoencoder.ckpt')],
			)

			SaveModelCallback(Path(model_dir) / 'lord-vc.pt').save_model(autoencoder)

	def convert(self, model_path: str, content_file_path: str, speaker_file_path: str, output_path: str):
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		wav2mel = Wav2Mel()
		mel2wav = Mel2Wav(sample_rate=wav2mel.sample_rate).to(device)

		model: AutoEncoder = torch.load(model_path, map_location=device).eval()

		with torch.no_grad():
			content_mel = wav2mel.parse_file(content_file_path).to(device)
			speaker_mel = wav2mel.parse_file(speaker_file_path).to(device)

			converted_mel = model.convert(
				content_img=content_mel[None, None, ...],
				class_img=speaker_mel[None, None, ...]
			)[0][0, 0]

			mel2wav.to_file(converted_mel, output_path)


if __name__ == '__main__':
	fire.Fire(Main())
