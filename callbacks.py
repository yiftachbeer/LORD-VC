import io
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import torch

import wandb


class GenerateSamplesCallback:

    def __init__(self, device):
        self.device = device

        self.visualized_imgs = []

    def generate_samples(self, latent_model, dataset, epoch, n_samples=4):
        latent_model.eval()
        with torch.no_grad():
            img_idx = torch.from_numpy(
                np.random.RandomState(seed=1234).choice(len(dataset), size=n_samples, replace=False).astype(np.int64))

            samples = dataset[img_idx]
            samples = {name: tensor.to(self.device) for name, tensor in samples.items()}
            fig = plt.figure(figsize=(10, 10))
            fig.suptitle(f'Step={epoch}')
            for i in range(n_samples):
                # Plot row headers (speaker)
                plt.subplot(n_samples + 1, n_samples + 1,
                            n_samples + 1 + i * (n_samples + 1) + 1)
                plt.imshow(samples['img'][i, 0].detach().cpu().numpy(), cmap='inferno')
                plt.gca().invert_yaxis()
                plt.axis('off')

                # Plot column headers (content)
                plt.subplot(n_samples + 1, n_samples + 1, i + 2)
                plt.imshow(samples['img'][i, 0].detach().cpu().numpy(), cmap='inferno')
                plt.gca().invert_yaxis()
                plt.axis('off')

                for j in range(n_samples):
                    plt.subplot(n_samples + 1, n_samples + 1,
                                n_samples + 2 + i * (n_samples + 1) + j + 1)

                    content_id = samples['img_id'][[j]]
                    class_id = samples['class_id'][[i]]
                    cvt = latent_model(content_id, class_id)['img'].squeeze().detach().cpu().numpy()

                    if epoch % 5 == 0:
                        np.savez(
                            f'samples/{epoch}_{content_id.item()}({samples["class_id"][[j]].item()})to{class_id.item()}.npz',
                            cvt)

                    plt.imshow(cvt, cmap='inferno')
                    plt.gca().invert_yaxis()
                    plt.axis('off')

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            pil_img = Image.open(buf)

            wandb.log({f'generated-{epoch}': [wandb.Image(pil_img)]}, step=epoch)
            self.visualized_imgs.append(np.asarray(pil_img).transpose(2, 0, 1)[:3])

            if epoch % 5 == 0:
                wandb.log({f'video': [
                    wandb.Video(np.array(self.visualized_imgs)),
                ]}, step=epoch)

    def generate_samples_amortized(self, amortized_model, dataset, epoch, n_samples=4):
        amortized_model.eval()

        with torch.no_grad():
            img_idx = torch.from_numpy(
                np.random.RandomState(seed=1234).choice(len(dataset), size=n_samples, replace=False).astype(np.int64))

            samples = dataset[img_idx]
            samples = {name: tensor.to(self.device) for name, tensor in samples.items()}
            fig = plt.figure(figsize=(10, 10))
            fig.suptitle(f'Step={epoch}')
            for i in range(n_samples):
                # Plot row headers (speaker)
                plt.subplot(n_samples + 1, n_samples + 1,
                            n_samples + 1 + i * (n_samples + 1) + 1)
                plt.imshow(samples['img'][i, 0].detach().cpu().numpy(), cmap='inferno')
                plt.gca().invert_yaxis()
                plt.axis('off')

                # Plot column headers (content)
                plt.subplot(n_samples + 1, n_samples + 1, i + 2)
                plt.imshow(samples['img'][i, 0].detach().cpu().numpy(), cmap='inferno')
                plt.gca().invert_yaxis()
                plt.axis('off')

                for j in range(n_samples):
                    plt.subplot(n_samples + 1, n_samples + 1,
                                n_samples + 2 + i * (n_samples + 1) + j + 1)

                    content_img = samples['img'][[j]]
                    class_img = samples['img'][[i]]
                    cvt = amortized_model.convert(content_img, class_img)['img'].squeeze().detach().cpu().numpy()

                    if epoch % 5 == 0:
                        np.savez(
                            f'samples/e{epoch}_{samples["img_id"][[j]].item()}({samples["class_id"][[j]].item()})to{samples["class_id"][[i]].item()}.npz',
                            cvt)

                    plt.imshow(cvt, cmap='inferno')
                    plt.gca().invert_yaxis()
                    plt.axis('off')

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            pil_img = Image.open(buf)

            wandb.log({f'generated-{epoch}': [wandb.Image(pil_img)]}, step=epoch)
            self.visualized_imgs.append(np.asarray(pil_img).transpose(2, 0, 1)[:3])

            if epoch % 5 == 0:
                wandb.log({f'video': [
                    wandb.Video(np.array(self.visualized_imgs)),
                ]}, step=epoch)