# biggan
[![jkyl](https://circleci.com/gh/jkyl/biggan.svg?style=shield)](https://app.circleci.com/pipelines/github/jkyl/biggan) [![PyPI version](https://badge.fury.io/py/biggan.svg)](https://badge.fury.io/py/biggan)

[BigGAN](https://arxiv.org/abs/1809.11096) in idiomatic [Keras](https://keras.io/about/).

### Install
```
pip install biggan
```
This installs the `biggan` library and top-level scripts. See [INSTALL.md](./INSTALL.md) to build from source.
### Prepare data
```
biggan.prepare --help
```
This serializes a set of image files into [TFRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord).
### Train
```
biggan.train --help
```
This trains a BigGAN model from scratch on the serialized image data. 
