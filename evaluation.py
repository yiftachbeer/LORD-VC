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


def tsne_plots(data_dir: str, model_path: str, segment: int = 128, n_utterances: int = 20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wav2mel = Wav2Mel()

    autoencoder: AutoEncoder = torch.load(model_path, map_location=device)
    autoencoder.eval()

    class_codes = []
    content_codes = []
    speaker_labels = []
    for speaker in tqdm(sorted(Path(data_dir).glob('*'))):
        for wav_file in sorted(speaker.rglob('*mic2.flac'))[:n_utterances]:
            with torch.no_grad():
                mel = wav2mel(*torchaudio.load(wav_file)).to(device)
                _, content_code, class_code = autoencoder(mel[None, None, ...])

                content_code = content_code[0].flatten().cpu().numpy()
                start = content_code.shape[0] // 2 - segment // 2
                content_code = content_code[start:start + segment]

                class_code = class_code[0].cpu().numpy()

                class_codes.append(class_code)
                content_codes.append(content_code)
                speaker_labels.append(speaker.name)

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    class_tsne = tsne.fit_transform(class_codes).T
    content_tsne = tsne.fit_transform(content_codes).T

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Class')
    sns.scatterplot(*class_tsne, hue=speaker_labels)

    plt.subplot(1, 2, 2)
    plt.title('Content')
    sns.scatterplot(*content_tsne, hue=speaker_labels)

    plt.show()


if __name__ == '__main__':
    fire.Fire(tsne_plots)
