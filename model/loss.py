import torch
from torch import nn
from torch import Tensor

from torchvision import models


class NetVGGFeatures(nn.Module):

    def __init__(self, layer_ids):
        super().__init__()

        self.vggnet = models.vgg16(pretrained=True)
        self.layer_ids = layer_ids

    def forward(self, x):
        output = []
        for i in range(self.layer_ids[-1] + 1):
            x = self.vggnet.features[i](x)

            if i in self.layer_ids:
                output.append(x)

        return output


class VGGDistance(nn.Module):

    def __init__(self, layer_ids):
        super().__init__()

        self.vgg = NetVGGFeatures(layer_ids)
        self.layer_ids = layer_ids

    def forward(self, I1, I2):
        # To apply VGG on grayscale, we duplicate the single channel
        I1 = torch.cat((I1, I1, I1), dim=1)
        I2 = torch.cat((I2, I2, I2), dim=1)

        b_sz = I1.size(0)
        f1 = self.vgg(I1)
        f2 = self.vgg(I2)

        loss = torch.abs(I1 - I2).view(b_sz, -1).mean(1)

        for i in range(len(self.layer_ids)):
            layer_loss = torch.abs(f1[i] - f2[i]).view(b_sz, -1).mean(1)
            loss = loss + layer_loss

        return loss.mean()


class SpeakerLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.dvector = torch.jit.load('pretrained/dvector.pt')
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        dvector_orig = self.dvector(x1.squeeze(1).transpose(1, 2))
        dvector_const = self.dvector(x2.squeeze(1).transpose(1, 2))
        return (1 - self.cos_sim(dvector_orig, dvector_const).mean()) / 2
