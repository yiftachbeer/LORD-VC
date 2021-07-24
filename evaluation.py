from pathlib import Path
from tqdm import tqdm
import fire
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

import torch
import torchaudio

from model.wav2mel import Wav2Mel
from model.adain_vc import AutoEncoder


def tsne_plots(data_dir, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wav2mel = Wav2Mel()

    autoencoder: AutoEncoder = torch.load(model_path, map_location=device)
    autoencoder.eval()

    class_codes = []
    content_codes = []
    speaker_labels = []
    for i_spk, spk in enumerate(tqdm(sorted(Path(data_dir).glob('*')))):
        for wav_file in sorted(spk.rglob('*mic2.flac')):
            mel = wav2mel(*torchaudio.load(wav_file))
            _, content_code, class_code = autoencoder(mel)

            class_codes.append(class_code.numpy())
            content_codes.append(content_code.numpy())
            speaker_labels.append(i_spk)

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    class_tsne = tsne.fit_transform(class_codes).T
    content_tsne = tsne.fit_transform(content_codes).T

    plt.subplot(1, 2, 1)
    sns.scatterplot(class_tsne, hue=speaker_labels)

    plt.subplot(1, 2, 2)
    sns.scatterplot(content_tsne, hue=speaker_labels)

    plt.show()


if __name__ == '__main__':
    fire.Fire(tsne_plots)
