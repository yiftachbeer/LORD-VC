import torch
from torch import nn
from torchvision import models

from model.adain_vc import SpeakerEncoder, ContentEncoder, Decoder


class LatentModel(nn.Module):

	def __init__(self, config):
		super().__init__()

		self.config = config

		self.content_embedding = RegularizedEmbedding(config['n_imgs'], config['content_dim'], config['content_std'])
		self.class_embedding = nn.Embedding(config['n_classes'], config['class_dim'])

		self.decoder = Decoder(**config['decoder_params'])

	def forward(self, img_id, class_id):
		content_code = self.content_embedding(img_id)
		class_code = self.class_embedding(class_id)

		# matching dims from LORD to AdaIN-VC decoder
		content_code = content_code.reshape((-1, 128, 16))

		generated_img = self.decoder(content_code, class_code).unsqueeze(1)  # add channel

		return {
			'img': generated_img,
			'content_code': content_code,
			'class_code': class_code
		}

	def init(self):
		self.apply(self.weights_init)

	@staticmethod
	def weights_init(m):
		if isinstance(m, nn.Embedding):
			nn.init.uniform_(m.weight, a=-0.05, b=0.05)


class AmortizedModel(nn.Module):

	def __init__(self, config):
		super().__init__()

		self.config = config

		self.content_encoder = ContentEncoder(**config['content_encoder_params'])
		self.class_encoder = SpeakerEncoder(**config['speaker_encoder_params'])
		self.decoder = Decoder(**config['decoder_params'])

	def forward(self, img):
		return self.convert(img, img)

	def convert(self, content_img, class_img):
		content_code = self.content_encoder(content_img.squeeze(1))
		class_code = self.class_encoder(class_img.squeeze(1))

		generated_img = self.decoder(content_code, class_code).unsqueeze(1)  # add channel

		return {
			'img': generated_img,
			'content_code': content_code,
			'class_code': class_code
		}


class RegularizedEmbedding(nn.Module):

	def __init__(self, num_embeddings, embedding_dim, stddev):
		super().__init__()

		self.embedding = nn.Embedding(num_embeddings, embedding_dim)
		self.stddev = stddev

	def forward(self, x):
		x = self.embedding(x)

		if self.training and self.stddev != 0:
			noise = torch.zeros_like(x)
			noise.normal_(mean=0, std=self.stddev)

			x = x + noise

		return x


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
