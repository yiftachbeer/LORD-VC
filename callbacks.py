from pathlib import Path
import io
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import wandb

import torch

from model.wav2mel import Mel2Wav


class TimedCallback:

    def __init__(self, callback, every: int = 1):
        self.callback = callback
        self.every = every

    def on_epoch_end(self, model, epoch):
        if (epoch + 1) % self.every == 0:
            self.callback.on_epoch_end(model, epoch)


class PlotTransferCallback:

    def __init__(self, dataset, n_samples=4, is_latent=True):
        self.dataset = dataset
        self.n_samples = n_samples
        self.is_latent = is_latent

        self.visualized_imgs = []

    def on_epoch_end(self, model, epoch):
        if self.is_latent:
            convert_fn = self.convert_latent
        else:
            convert_fn = self.convert_autoencoder

        self.generate_plot(model, epoch, convert_fn)

    def convert_latent(self, model, i, j, imgs, img_ids, class_ids):
        content_id = img_ids[[j]]
        class_id = class_ids[[i]]
        return model(content_id, class_id)

    def convert_autoencoder(self, model, i, j, imgs, img_ids, class_ids):
        content_img = imgs[[j]]
        class_img = imgs[[i]]
        return model.convert(content_img, class_img)

    def generate_plot(self, model, epoch, convert_fn):
        device = next(model.parameters()).device

        model.eval()
        with torch.no_grad():
            img_idx = torch.from_numpy(
                np.random.RandomState(seed=4).choice(len(self.dataset), size=self.n_samples, replace=False).astype(
                    np.int64))

            img_ids, class_ids, imgs = [tensor.to(device) for tensor in self.dataset[img_idx]]
            grid_to_plot = [None] * ((self.n_samples + 1) * (self.n_samples + 1))
            for i in range(self.n_samples):
                # row headers (class)
                grid_to_plot[(i + 1) * (self.n_samples + 1)] = imgs[i, 0].detach().cpu().numpy()

                # column headers (content)
                grid_to_plot[i + 1] = imgs[i, 0].detach().cpu().numpy()
                for j in range(self.n_samples):
                    # converted image with class i and content j
                    converted = convert_fn(model, i, j, imgs, img_ids.squeeze(0), class_ids)[0].squeeze(0).detach().cpu().numpy()
                    grid_to_plot[(self.n_samples + 2) + i * (self.n_samples + 1) + j] = converted

            fig = plt.figure()
            grid = ImageGrid(fig, 111, nrows_ncols=(self.n_samples + 1, self.n_samples + 1))
            for ax, img in zip(grid, grid_to_plot):
                ax.axis('off')
                if img is None:
                    # first cell
                    ax.set_title(f'Step={epoch}')
                else:
                    ax.imshow(img, cmap='inferno')
                    ax.invert_yaxis()
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            pil_img = Image.open(buf)

            wandb.log({f'transfer plot-{epoch:03}': [wandb.Image(pil_img)]}, step=epoch)
            self.visualized_imgs.append(np.asarray(pil_img).transpose(2, 0, 1)[:3])

            if epoch % 5 == 0:
                wandb.log({f'video': [
                    wandb.Video(np.array(self.visualized_imgs)),
                ]}, step=epoch)


class GenerateAudioSamplesCallback:

    def __init__(self, dataset, save_dir: Path, n_samples=3, save_every: int = 5, is_latent=True):
        self.dataset = dataset
        self.save_dir = save_dir
        self.mel2wav = Mel2Wav()
        self.n_samples = n_samples
        self.save_every = save_every

        self.is_latent = is_latent

    def convert_latent(self, model, i, j, imgs, img_ids, class_ids):
        content_id = img_ids[[j]]
        class_id = class_ids[[i]]
        return model(content_id, class_id)

    def convert_autoencoder(self, model, i, j, imgs, img_ids, class_ids):
        content_img = imgs[[j]]
        class_img = imgs[[i]]
        return model.convert(content_img, class_img)

    def on_epoch_end(self, model, epoch):
        if self.is_latent:
            convert_fn = self.convert_latent
        else:
            convert_fn = self.convert_autoencoder

        self.save_samples(model, epoch, convert_fn)

    def save_samples(self, model, epoch, convert_fn):
        device = next(model.parameters()).device

        model.eval()
        with torch.no_grad():
            img_idx = torch.from_numpy(
                np.random.RandomState(seed=4).choice(len(self.dataset), size=self.n_samples, replace=False).astype(
                    np.int64))

            img_ids, class_ids, imgs = [tensor.to(device) for tensor in self.dataset[img_idx]]

            mels = []
            paths = []

            samples_dir = Path(self.save_dir / f'{epoch:03}')
            if not samples_dir.exists():
                samples_dir.mkdir(parents=True)

            for i in range(self.n_samples):
                for j in range(self.n_samples):
                    converted = convert_fn(model, i, j, imgs.squeeze(1), img_ids, class_ids)[0][0]
                    mels.append(converted)

                    content_id = img_ids[j]
                    orig_class_id = class_ids[j]
                    converted_class_id = class_ids[i]
                    if orig_class_id == converted_class_id:
                        paths.append(samples_dir / f'recons_{content_id}({orig_class_id}).wav')
                    else:
                        paths.append(samples_dir / f'transfer_{content_id}({orig_class_id}to{converted_class_id}).wav')

            model.cpu()
            self.mel2wav.to(device)

            self.mel2wav.to_files(mels, paths)
            wandb.log({f'samples-{epoch:03}': [wandb.Audio(str(path), caption=path.name) for path in paths]}, step=epoch)

            self.mel2wav.cpu()
            model.to(device)


class SaveCheckpointCallback:

    def __init__(self, path_to_save: Path):
        self.path_to_save = str(path_to_save)

        if not path_to_save.parent.exists():
            path_to_save.parent.mkdir(parents=True)

    def on_epoch_end(self, model, epoch):
        torch.save(model.state_dict(), self.path_to_save)
        wandb.save(self.path_to_save)


class SaveModelCallback:

    def __init__(self, path_to_save: Path):
        self.path_to_save = str(path_to_save)

        if not path_to_save.parent.exists():
            path_to_save.parent.mkdir(parents=True)

    def save_model(self, model, suffix=''):
        path = self.path_to_save + suffix + '.pt'

        jit_model = torch.jit.script(model)
        jit_model.save(path)
        wandb.save(path)

    def on_epoch_end(self, model, epoch):
        self.save_model(model, suffix=f'-{epoch}')
