# biggan
[BigGAN](https://arxiv.org/abs/1809.11096) in idiomatic Keras.

### Setup
```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
poetry build && poetry install
```
### Prepare data
```
poetry run biggan.prepare
```
### Train
```
poetry run biggan.train
```
