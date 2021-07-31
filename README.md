# LORD-VC
Disentanglement of speaker from speech by combining [LORD](http://www.vision.huji.ac.il/lord) and [AdaIN-VC](https://arxiv.org/abs/1904.05742).

## Usage

### Training

Preprocessing a dataset:
```
main.py preprocess <data_dir> <save_dest> [<segment>]
```

Training a model with latent optimization (first stage):
```
main.py train <data_path> <save_path>
```

Training the autoencoder (second stage):
```
main.py train_encoders <data_path> <model_dir>
```

Converting the speaker of an audio sample to that of another using a trained model:
```
main.py convert <model_path> <content_file_path> <speaker_file_path> <output_path>
```

### Evaluation

Creating t-SNE plots for class and content:
```
evaluation.py tsne <data_dir> <model_path> [<segment>] [<n_utterances>]
```

Calculating neural MOS scores for generated samples:
```
evaluation.py mean_opinion_score <data_dir>
```

## See Also

* [Demystifying Inter-Class Disentanglement](http://www.vision.huji.ac.il/lord)
* [One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization](https://arxiv.org/abs/1904.05742)