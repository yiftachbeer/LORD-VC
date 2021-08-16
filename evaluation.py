from pathlib import Path
import fire
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

import librosa
import sox
from resemblyzer import preprocess_wav, VoiceEncoder

import torch

from model.wav2mel import Wav2Mel
from model.lord import AutoEncoder


def mean_opinion_score(data_path: str, pretrained_path: str = 'pretrained/neural_mos.pt'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    neural_mos = torch.jit.load(pretrained_path, map_location=device).eval()

    tfm = sox.Transformer()

    scores = []
    for file_path in Path(data_path).glob('*'):
        wav, _ = librosa.load(file_path, sr=16000)
        wav = tfm.build_array(input_array=wav, sample_rate_in=16000)
        spect = np.abs(librosa.stft(wav, n_fft=512)).T[None, None, ...]
        spect = torch.from_numpy(spect)

        with torch.no_grad():
            spect = spect.to(device)
            score = neural_mos.only_mean_inference(spect)
            scores.append(score.item())

    return np.mean(scores)


def speaker_verification(converted_files_dir: str, speakers_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resemblyzer = VoiceEncoder(device, verbose=False)

    cosine_similarities = []
    speaker_embeddings = {}
    for converted_path in Path(converted_files_dir).glob('*'):
        speaker_name = converted_path.name.split('_')[0]
        if speaker_name not in speaker_embeddings:
            speaker_wavs = [preprocess_wav(path) for path in (Path(speakers_dir) / speaker_name).glob('*')]
            speaker_embedding = resemblyzer.embed_speaker(speaker_wavs)
            speaker_embeddings[speaker_name] = speaker_embedding
        else:
            speaker_embedding = speaker_embeddings[speaker_name]

        converted_speaker_embedding = resemblyzer.embed_utterance(preprocess_wav(converted_path))

        cosine_similarity = (
                np.inner(converted_speaker_embedding, speaker_embedding) / np.linalg.norm(converted_speaker_embedding) / np.linalg.norm(speaker_embedding)
        )
        cosine_similarities.append(cosine_similarity)

    return np.mean(cosine_similarities), cosine_similarities


def tsne_plots(data_dir: str, model_path: str, segment: int = 128, n_utterances: int = 20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wav2mel = Wav2Mel()

    autoencoder: AutoEncoder = torch.jit.load(model_path, map_location=device).eval()

    class_codes = []
    content_codes = []
    speaker_labels = []
    for speaker in sorted(Path(data_dir).glob('*')):
        for wav_file in sorted(speaker.rglob('*mic2.flac'))[:n_utterances]:
            with torch.no_grad():
                mel = wav2mel.parse_file(wav_file).to(device)
                _, content_code, class_code = autoencoder(mel[None, None, ...])

                content_code = content_code[0].flatten().cpu().numpy()
                start = content_code.shape[0] // 2 - segment // 2
                content_code = content_code[start:start + segment]

                class_code = class_code[0].cpu().numpy()

                class_codes.append(class_code)
                content_codes.append(content_code)
                speaker_labels.append(speaker.name)

    tsne = TSNE(n_components=2, perplexity=25, n_iter=300)
    class_x, class_y = tsne.fit_transform(class_codes).T
    content_x, content_y = tsne.fit_transform(content_codes).T

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Class')
    sns.scatterplot(x=class_x, y=class_y, hue=speaker_labels)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Content')
    sns.scatterplot(x=content_x, y=content_y, hue=speaker_labels)
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    fire.Fire({
        'tsne': tsne_plots,
        'mos': mean_opinion_score,
        'speaker_verification': speaker_verification,
    })
