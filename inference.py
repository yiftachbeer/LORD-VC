from pathlib import Path
import fire

import torch

from model.wav2mel import Wav2Mel, Mel2Wav
from model.lord import AutoEncoder


def _convert_pair(model, content_file_path, speaker_file_path, wav2mel, device):
    with torch.no_grad():
        content_mel = wav2mel.parse_file(content_file_path).to(device)
        speaker_mel = wav2mel.parse_file(speaker_file_path).to(device)

        return model.convert(content_mel[None, ...], speaker_mel[None, ...])[0].squeeze(0)


def convert(model_path: str, content_file_path: str, speaker_file_path: str, output_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wav2mel = Wav2Mel()
    mel2wav = Mel2Wav(sample_rate=wav2mel.sample_rate).to(device)

    model: AutoEncoder = torch.jit.load(model_path, map_location=device).eval()

    converted_mel = _convert_pair(model, content_file_path, speaker_file_path, wav2mel, device)

    with torch.no_grad():
        mel2wav.to_file(converted_mel, output_path)


def convert_many(model_path: str, pairs_file_path: str, output_dir: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wav2mel = Wav2Mel()
    mel2wav = Mel2Wav(sample_rate=wav2mel.sample_rate).to(device)

    model: AutoEncoder = torch.jit.load(model_path, map_location=device).eval()

    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True)

    converted_mels = []
    output_paths = []
    with open(pairs_file_path, 'r') as f:
        for line in f.readlines():
            content_file_path, speaker_file_path = line.split()
            speaker_name = Path(speaker_file_path).name.split('_', 1)[0]
            rest_of_name = Path(content_file_path).name.split('_', 1)[1]
            output_paths.append(str(output_dir_path / f'{speaker_name}_{rest_of_name}'))

            converted_mel = _convert_pair(model, content_file_path, speaker_file_path, wav2mel, device)
            converted_mels.append(converted_mel)

    with torch.no_grad():
        mel2wav.to_files(converted_mels, output_paths)


if __name__ == '__main__':
    fire.Fire({
        'convert': convert,
        'convert_many': convert_many,
    })
