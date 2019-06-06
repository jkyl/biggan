# BigGAN-deep
A [BigGAN](https://arxiv.org/abs/1809.11096) training algorithm for more modest hardware

## Setup
```bash
# download dependencies
$ pip install tensorflow-gpu==2.0.0a0 opencv-python

# preprocess some image data offline
$ python src/data.py /nested/path/to/images/ output.npy
```

## Usage
```
$ python main.py --help
usage: main.py [-h] [-bs BATCH_SIZE] [-ch CHANNELS] data_file model_dir

positional arguments:
  data_file       .npy file containing preprocessed image data
  model_dir       directory in which to save checkpoints and summaries

optional arguments:
  -h, --help      show this help message and exit
  -bs BATCH_SIZE  number of samples per minibatch update (default: 64)
  -ch CHANNELS    channel multiplier in G and D (default: 16)
```
