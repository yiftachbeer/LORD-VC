default_config = dict(
	content_dim=2048,
	class_dim=128,

	content_std=1,
	content_decay=1e-4,
	lambda_speaker=1,

	content_encoder_params=dict(
		c_in=80,
		c_h=128,
		c_out=128,
		kernel_size=5,
		bank_size=8,
		bank_scale=1,
		c_bank=128,
		n_conv_blocks=6,
		subsample=[1, 2, 1, 2, 1, 2],
		act="relu",
		dropout_rate=0.0
	),

	speaker_encoder_params=dict(
		c_in=80,
		c_h=128,
		c_out=128,
		kernel_size=5,
		bank_size=8,
		bank_scale=1,
		c_bank=128,
		n_conv_blocks=6,
		n_dense_blocks=6,
		subsample=[1, 2, 1, 2, 1, 2],
		act="relu",
		dropout_rate=0.0
	),

	decoder_params=dict(
		c_in=128,
		c_cond=128,
		c_h=128,
		c_out=80,
		kernel_size=5,
		n_conv_blocks=6,
		upsample=[2, 1, 2, 1, 2, 1],
		act="relu",
		sn=False,
		dropout_rate=0.0
	),

	perceptual_loss=dict(
		layers=[2, 7, 12, 21, 30]
	),

	train=dict(
		batch_size=128,
		n_epochs=200,

		learning_rate=dict(
			latent=3e-2,
			decoder=3e-3,
			min=1e-4
		)
	),

	train_encoders=dict(
		batch_size=128,
		n_epochs=200,

		learning_rate=dict(
			max=1e-4,
			min=1e-5
		)
	),
)


def update_nested(d1: dict, d2: dict):
	"""
	Update d1, that might have nested dictionaries with the items of d2.
	Nested keys in d2 should be separated with a '/' character.
	Example:
	>> d1 = {'a': {'b': 1, 'c': 2}}
	>> d2 = {'a/b': 3, 'd': 4}
	>> update_nested(d1, d2)
	d1 = {'a': {'b': 3, 'c': 2}, 'd': 4}

	:param d1: The dict to update.
	:param d2: The values to update with.
	:return: The updated dict.
	"""
	for key, val in d2.items():
		key_path = key.split('/')
		curr_d = d1
		for key in key_path[:-1]:
			curr_d = curr_d[key]
		curr_d[key_path[-1]] = val

	return d1


def get_config(**kwargs):
	config = default_config

	update_nested(config, kwargs)

	return config
