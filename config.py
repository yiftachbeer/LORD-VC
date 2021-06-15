base_config = dict(
	content_dim=2048,
	class_dim=128,


	content_std=1,
	content_decay=1e-4,

	n_adain_layers=4,
	adain_dim=256,

	# decoder params
	c_in=128,
	c_cond=128,
	c_h=128,
	c_out=80,
	kernel_size=5,
	n_conv_blocks=6,
	upsample=[2, 1, 2, 1, 2, 1],
	act="relu",
	sn=False,
	dropout_rate=0.0,

	perceptual_loss=dict(
		layers=[2, 7, 12, 21, 30]
	),

	train=dict(
		batch_size=16,
		n_epochs=200,

		learning_rate=dict(
			generator=3e-3,
			latent=3e-2,
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
