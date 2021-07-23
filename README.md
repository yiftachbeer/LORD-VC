# LORD-VC
An adaptation of LORD ([Demystifying Inter-Class Disentanglement](http://www.vision.huji.ac.il/lord)) to speech audio.
Also borrows heavily from ([One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization](https://arxiv.org/abs/1904.05742)).

### Getting started
Training a model for disentanglement requires several steps.

#### Preprocessing an image dataset
Preprocessing a local copy of one of the supported datasets can be done as follows:
```
lord.py --base-dir <output-root-dir> preprocess
    --dataset-id {mnist,smallnorb,cars3d,shapes3d,celeba,kth,rafd}
    --dataset-path <input-dataset-path>
    --data-name <output-data-filename>
```

Splitting a preprocessed dataset into train and test sets can be done according to one of two configurations:
```
lord.py --base-dir <output-root-dir> split-classes
    --input-data-name <input-data-filename>
    --train-data-name <output-train-data-filename>
    --test-data-name <output-test-data-filename>
    --num-test-classes <number-of-random-test-classes>
```

```
lord.py --base-dir <output-root-dir> split-samples
    --input-data-name <input-data-filename>
    --train-data-name <output-train-data-filename>
    --test-data-name <output-test-data-filename>
    --test-split <ratio-of-random-test-samples>
```

#### Training a model
Given a preprocessed train set, training a model with latent optimization (first stage) can be done as follows:
```
lord.py --base-dir <output-root-dir> train
    --data-name <input-preprocessed-data-filename>
    --model-name <output-model-name>
```

Training encoders for amortized inference (second stage) can be done as follows:
```
lord.py --base-dir <output-root-dir> train-encoders
    --data-name <input-preprocessed-data-filename>
    --model-name <input-model-name>
```

