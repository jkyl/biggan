# BigGAN-deep
A [BigGAN](https://arxiv.org/abs/1809.11096) training algorithm for modest hardware.

## Setup with [Poetry](https://python-poetry.org/):
```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
poetry build && poetry install
```
## Prepare data:
```
poetry run biggan prepare
```
## Train:
```poetry run biggan train```
```
usage: train [-h] [-bs BATCH_SIZE] [-ch CHANNELS] data_file model_dir

positional arguments:
  data_file       .npz file containing labeled image data
  model_dir       directory in which to save checkpoints and summaries

optional arguments:
  -h, --help      show this help message and exit
  -bs BATCH_SIZE  number of samples per minibatch update (default: 64)
  -ch CHANNELS    channel multiplier in G and D (default: 48)
```
