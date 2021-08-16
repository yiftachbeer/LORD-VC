# LORD-VC
Disentanglement of speaker from speech by combining [LORD](http://www.vision.huji.ac.il/lord) and [AdaIN-VC](https://arxiv.org/abs/1904.05742).

## Usage

For all demonstrated files, using `--help` shows documentation both at file level and command level (e.g. `main.py --help` lists commands, `main.py train --help` details available flags).

For training, any key in the dictionary in `config.py` can be overridden. Nested keys can be accessed using a `/`, e.g. `--train_encoders/n_epochs 200`.   

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

### Inference

Converting the speaker of an audio sample to that of another using a trained model:
```
inference.py convert <model_path> <content_file_path> <speaker_file_path> <output_path>
```

Multiple conversions can be performed at once by creating a file with lines of the form `<content_file_path> <speaker_file_path> <output_path>` which we call a pairs file:
```
inference.py convert_many <model_path> <pairs_file_path>
```

### Evaluation

Calculating neural MOS scores for generated samples:
```
evaluation.py mean_opinion_score <data_dir>
```

Evaluate the similarity between the speakers in converted files and their original recordings:

```
evaluation.py speaker_verification <converted_files_dir> <speakers_dir>
```

Creating t-SNE plots for class and content:
```
evaluation.py tsne <data_dir> <model_path> [<segment>] [<n_utterances>]
```


## See Also

* [Demystifying Inter-Class Disentanglement](http://www.vision.huji.ac.il/lord)
* [One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization](https://arxiv.org/abs/1904.05742)