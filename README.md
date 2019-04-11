# generative
a "big gan" training algorithm for modest-ish hardware

## Usage
```
$ python main.py --help
usage: main.py [-h] [-bs BATCH_SIZE] [-ch CHANNELS] data_file model_dir

positional arguments:
  data_file       .npz file containing preprocessed image data
  model_dir       directory in which to save checkpoints and summaries

optional arguments:
  -h, --help      show this help message and exit
  -bs BATCH_SIZE  number of samples per minibatch update (default: 64)
  -ch CHANNELS    channel multiplier in G and D (default: 16)
```
