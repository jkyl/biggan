# BigGAN-deep
A [BigGAN](https://arxiv.org/abs/1809.11096) training algorithm for modest<sup>†</sup> hardware
> †: tested on 4x Titan Xp and 8x V100, both running TensorFlow 2.0

## Setup
```bash
# download dependencies
$ pip install numpy==1.17.2 tensorflow-gpu==2.0.0 opencv-python joblib

# preprocess some image data offline
$ python src/data.py /nested/path/to/images/ output.npz
```

## Usage
```
$ python main.py --help
usage: main.py [-h] [-bs BATCH_SIZE] [-ch CHANNELS] data_file model_dir

positional arguments:
  data_file       .npz file containing labeled image data
  model_dir       directory in which to save checkpoints and summaries

optional arguments:
  -h, --help      show this help message and exit
  -bs BATCH_SIZE  number of samples per minibatch update (default: 64)
  -ch CHANNELS    channel multiplier in G and D (default: 48)
```
