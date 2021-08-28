import warnings
import pickle
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
from torch import nn

from audio import Wav2Mel
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

            with warnings.catch_warnings():
                # A patch to remove the following message:
                # "UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they
                # need to be compacted at every call, possibly greately increasing memory usage. To compact weights
                # again call flatten_parameters()."
                warnings.simplefilter("ignore")

                score = neural_mos.only_mean_inference(spect)
            scores.append(score.item())

    return np.mean(scores)


def _create_speaker_embeddings(resemblyzer, speakers_dir: str, save_path: str = None):
    speaker_embeddings = {}
    for speaker_path in Path(speakers_dir).glob('*'):
        speaker_name = speaker_path.name
        speaker_wavs = [preprocess_wav(path) for path in (Path(speakers_dir) / speaker_name).glob('*')]
        speaker_embeddings[speaker_name] = resemblyzer.embed_speaker(speaker_wavs)

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(speaker_embeddings, f)

    return speaker_embeddings


def speaker_verification(converted_files_dir: str, speakers_dir: str, speaker_embeddings_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resemblyzer = VoiceEncoder(device, verbose=False)

    # get speaker embeddings
    if speaker_embeddings_path and Path(speaker_embeddings_path).exists():
        with open(speaker_embeddings_path, 'rb') as f:
            speaker_embeddings = pickle.load(f)
    else:
        speaker_embeddings = _create_speaker_embeddings(resemblyzer, speakers_dir, speaker_embeddings_path)

    speaker_names = list(speaker_embeddings.keys())
    speaker_name2idx = {name: i for i, name in enumerate(speaker_names)}
    speaker_embeddings_tensor = torch.from_numpy(np.array(list(speaker_embeddings.values())))

    # calculate similarities
    similarity = nn.CosineSimilarity()
    successes = []
    for converted_path in Path(converted_files_dir).glob('*'):
        speaker_name = converted_path.name.split('_')[0]
        speaker_idx = speaker_name2idx[speaker_name]
        converted_speaker_embedding = torch.from_numpy(resemblyzer.embed_utterance(preprocess_wav(converted_path)))

        cosine_similarities = similarity(converted_speaker_embedding[None, :], speaker_embeddings_tensor)
        success = torch.argmax(cosine_similarities) == speaker_idx
        successes.append(success)

    return torch.tensor(successes).float().mean().item()


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
                _, content_code, class_code = autoencoder(mel[None, ...])

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
