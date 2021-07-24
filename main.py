from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import fire
import wandb
import soundfile as sf

import torch
import torchaudio

from data import get_data, get_dataloader, get_latent_codes_dataloader
from training import train_latent, train_amortized
from config import get_config, save_config
from model.wav2mel import Wav2Mel
from model.adain_vc import get_latent_model, get_autoencoder
from callbacks import GenerateSamplesCallback, SaveModelCallback


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
		dataset, imgs, data = get_data(data_path)
		config = get_config(
			img_shape=imgs.shape[1:],
			n_imgs=imgs.shape[0],
			n_classes=data['n_classes'].item(),
			kwargs=kwargs
		)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		model = get_latent_model(config)
		model.init()
		model.to(device)

		data_loader = get_dataloader(dataset, config['train']['batch_size'], device)

		with wandb.init(config=config):
			save_config(config, Path(save_path) / 'config.pkl')
			train_latent(
				model=model,
				config=config,
				device=device,
				data_loader=data_loader,
				callbacks=[GenerateSamplesCallback(device, dataset, is_latent=True),
						   SaveModelCallback(str(Path(save_path) / 'latent.ckpt'))],
			)

	def train_encoders(self, data_path: str, model_dir: str, **kwargs):
		dataset, imgs, data = get_data(data_path)
		config = get_config(
			img_shape=imgs.shape[1:],
			n_imgs=imgs.shape[0],
			n_classes=data['n_classes'].item(),
			kwargs=kwargs
		)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		latent_model = get_latent_model(config)
		latent_model.load_state_dict(torch.load(Path(model_dir) / 'latent.ckpt'))
		latent_model.to(device)
		latent_model.eval()

		autoencoder = get_autoencoder(config)
		autoencoder.to(device)
		autoencoder.decoder.load_state_dict(latent_model.decoder.state_dict())

		data_loader = get_latent_codes_dataloader(dataset, config['train']['batch_size'], device, latent_model)

		with wandb.init(config=config):
			save_config(config, Path(model_dir) / 'config.pkl')
			train_amortized(
				model=autoencoder,
				config=config,
				device=device,
				data_loader=data_loader,
				callbacks=[GenerateSamplesCallback(device, dataset, is_latent=False),
						   SaveModelCallback(str(Path(model_dir) / 'autoencoder.ckpt'))],
			)

		torch.save(autoencoder, str(Path(model_dir) / 'lord-vc.pt'))

	def convert(self, model_path: str, content_file_path: str, speaker_file_path: str, output_path: str,
				vocoder_path: str = r"pretrained\vocoder.pth"):
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		wav2mel = Wav2Mel()

		model = torch.load(model_path, map_location=device)
		model.eval()

		vocoder = torch.jit.load(vocoder_path, map_location=device)
		vocoder.eval()

		with torch.no_grad():
			content_mel = wav2mel(*torchaudio.load(content_file_path)).to(device)
			speaker_mel = wav2mel(*torchaudio.load(speaker_file_path)).to(device)

			converted_mel = model.convert(
				content_img=content_mel[None, None, ...],
				class_img=speaker_mel[None, None, ...]
			)[0]

			wav = vocoder.generate([converted_mel[0, 0].T])[0]
			sf.write(output_path, wav.data.cpu().numpy(), wav2mel.sample_rate)


if __name__ == '__main__':
	fire.Fire(Main())
