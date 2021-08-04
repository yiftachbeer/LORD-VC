import torch

from model.wav2mel import Wav2Mel, Mel2Wav
from model.lord import AutoEncoder


def _convert_pair(model, content_file_path, speaker_file_path, wav2mel, device):
    with torch.no_grad():
        content_mel = wav2mel.parse_file(content_file_path).to(device)
        speaker_mel = wav2mel.parse_file(speaker_file_path).to(device)

        return model.convert(
            content_img=content_mel[None, None, ...],
            class_img=speaker_mel[None, None, ...]
        )[0][0, 0]


def convert(model_path: str, content_file_path: str, speaker_file_path: str, output_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wav2mel = Wav2Mel()
    mel2wav = Mel2Wav(sample_rate=wav2mel.sample_rate).to(device)

    model: AutoEncoder = torch.load(model_path, map_location=device).eval()

    converted_mel = _convert_pair(model, content_file_path, speaker_file_path, wav2mel, device)
    mel2wav.to_file(converted_mel, output_path)


def convert_many(model_path: str, pairs_file_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wav2mel = Wav2Mel()
    mel2wav = Mel2Wav(sample_rate=wav2mel.sample_rate).to(device)

    model: AutoEncoder = torch.load(model_path, map_location=device).eval()

    converted_mels = []
    output_paths = []
    with open(pairs_file_path, 'rb') as f:
        for line in f.readlines():
            content_file_path, speaker_file_path, output_path = line.split()
            output_paths.append(output_path)

            converted_mel = _convert_pair(model, content_file_path, speaker_file_path, wav2mel, device)
            converted_mels.append(converted_mel)

    mel2wav.to_files(converted_mels, output_paths)

