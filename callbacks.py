from pathlib import Path
import io
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import torch

import wandb


class GenerateSamplesCallback:

    def __init__(self, device, dataset, n_samples=4, is_latent=True):
        self.device = device
        self.dataset = dataset
        self.n_samples = n_samples
        self.is_latent = is_latent

        self.visualized_imgs = []

    def on_epoch_end(self, model, epoch):
        if self.is_latent:
            convert_fn = self.convert_latent
        else:
            convert_fn = self.convert_amortized

        self.save_sample(model, epoch, convert_fn)

    def convert_latent(self, model, i, j, imgs, img_ids, class_ids):
        content_id = img_ids[[j]]
        class_id = class_ids[[i]]
        return model(content_id, class_id)

    def convert_amortized(self, model, i, j, imgs, img_ids, class_ids):
        content_img = imgs[[j]]
        class_img = imgs[[i]]
        return model.convert(content_img, class_img)

    def save_sample(self, model, epoch, convert_fn):
        model.eval()
        with torch.no_grad():
            img_idx = torch.from_numpy(
                np.random.RandomState(seed=1234).choice(len(self.dataset), size=self.n_samples, replace=False).astype(
                    np.int64))

            samples = self.dataset[img_idx]
            img_ids, class_ids, imgs = [tensor.to(self.device) for tensor in samples]
            grid_to_plot = [None] * ((self.n_samples + 1) * (self.n_samples + 1))
            for i in range(self.n_samples):
                # row headers (class)
                grid_to_plot[(i + 1) * (self.n_samples + 1)] = imgs[i, 0].detach().cpu().numpy()

                # column headers (content)
                grid_to_plot[i + 1] = imgs[i, 0].detach().cpu().numpy()
                for j in range(self.n_samples):
                    # converted image with class i and content j
                    converted = convert_fn(model, i, j, imgs, img_ids, class_ids)[0].squeeze().detach().cpu().numpy()
                    grid_to_plot[(self.n_samples + 2) + i * (self.n_samples + 1) + j] = converted

                    if epoch % 5 == 0:
                        # Also save result to file
                        content_id = img_ids[j].item()
                        orig_class_id = class_ids[j].item()
                        converted_class_id = class_ids[i].item()
                        np.savez(f'samples/{epoch}_{content_id}({orig_class_id})to{converted_class_id}.npz', converted)

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

            wandb.log({f'generated-{epoch}': [wandb.Image(pil_img)]}, step=epoch)
            self.visualized_imgs.append(np.asarray(pil_img).transpose(2, 0, 1)[:3])

            if epoch % 5 == 0:
                wandb.log({f'video': [
                    wandb.Video(np.array(self.visualized_imgs)),
                ]}, step=epoch)


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

    def on_epoch_end(self, model, epoch):
        torch.save(model, self.path_to_save)
        wandb.save(self.path_to_save)
