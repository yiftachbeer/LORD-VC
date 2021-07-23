import io
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import torch

import wandb


class GenerateSamplesLatentCallback:

    def __init__(self, device, dataset, n_samples=4):
        self.device = device
        self.dataset = dataset
        self.n_samples = n_samples

        self.visualized_imgs = []

    def on_epoch_end(self, latent_model, epoch):
        latent_model.eval()
        with torch.no_grad():
            img_idx = torch.from_numpy(
                np.random.RandomState(seed=1234).choice(len(self.dataset), size=self.n_samples, replace=False).astype(np.int64))

            samples = self.dataset[img_idx]
            samples = {name: tensor.to(self.device) for name, tensor in samples.items()}
            fig = plt.figure(figsize=(10, 10))
            fig.suptitle(f'Step={epoch}')
            for i in range(self.n_samples):
                # Plot row headers (speaker)
                plt.subplot(self.n_samples + 1, self.n_samples + 1,
                            self.n_samples + 1 + i * (self.n_samples + 1) + 1)
                plt.imshow(samples['img'][i, 0].detach().cpu().numpy(), cmap='inferno')
                plt.gca().invert_yaxis()
                plt.axis('off')

                # Plot column headers (content)
                plt.subplot(self.n_samples + 1, self.n_samples + 1, i + 2)
                plt.imshow(samples['img'][i, 0].detach().cpu().numpy(), cmap='inferno')
                plt.gca().invert_yaxis()
                plt.axis('off')

                for j in range(self.n_samples):
                    plt.subplot(self.n_samples + 1, self.n_samples + 1,
                                self.n_samples + 2 + i * (self.n_samples + 1) + j + 1)

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


class GenerateSamplesAmortizedCallback:

    def __init__(self, device, dataset, n_samples=4):
        self.device = device
        self.dataset = dataset
        self.n_samples = n_samples

        self.visualized_imgs = []

    def on_epoch_end(self, amortized_model, epoch):
        amortized_model.eval()

        with torch.no_grad():
            img_idx = torch.from_numpy(
                np.random.RandomState(seed=1234).choice(len(self.dataset), size=self.n_samples, replace=False).astype(np.int64))

            samples = self.dataset[img_idx]
            samples = {name: tensor.to(self.device) for name, tensor in samples.items()}
            fig = plt.figure(figsize=(10, 10))
            fig.suptitle(f'Step={epoch}')
            for i in range(self.n_samples):
                # Plot row headers (speaker)
                plt.subplot(self.n_samples + 1, self.n_samples + 1,
                            self.n_samples + 1 + i * (self.n_samples + 1) + 1)
                plt.imshow(samples['img'][i, 0].detach().cpu().numpy(), cmap='inferno')
                plt.gca().invert_yaxis()
                plt.axis('off')

                # Plot column headers (content)
                plt.subplot(self.n_samples + 1, self.n_samples + 1, i + 2)
                plt.imshow(samples['img'][i, 0].detach().cpu().numpy(), cmap='inferno')
                plt.gca().invert_yaxis()
                plt.axis('off')

                for j in range(self.n_samples):
                    plt.subplot(self.n_samples + 1, self.n_samples + 1,
                                self.n_samples + 2 + i * (self.n_samples + 1) + j + 1)

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


class SaveModelCallback:

    def __init__(self, path_to_save):
        self.path_to_save = path_to_save

    def on_epoch_end(self, model, epoch):
        torch.save(model.state_dict(), self.path_to_save)
        wandb.save(self.path_to_save)