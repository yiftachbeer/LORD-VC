import argparse
import os
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import torch.nn as nn
import torchaudio
from torch import Tensor

from model.wav2mel import Wav2Mel


def process_files(audio_file: str, wav2mel: nn.Module) -> Tensor:
    speech_tensor, sample_rate = torchaudio.load(audio_file)
    mel_tensor = wav2mel(speech_tensor, sample_rate)

    return mel_tensor


def main(data_dir: str, save_dest: str, segment: int = 128):
    wav2mel = Wav2Mel()

    cropped_mels = []
    classes = []
    file_names = []

    for i_spk, spk in enumerate(tqdm(sorted(os.listdir(data_dir)))):
        for wav_file in sorted((Path(data_dir) / spk).rglob('*mic2.flac')):
            mel = process_files(wav_file, wav2mel)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("save_dest", type=str)
    parser.add_argument("--segment", type=int, default=128)
    main(**vars(parser.parse_args()))
