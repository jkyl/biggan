# biggan
[![jkyl](https://circleci.com/gh/jkyl/biggan.svg?style=shield)](https://app.circleci.com/pipelines/github/jkyl/biggan)

[BigGAN](https://arxiv.org/abs/1809.11096) in idiomatic Keras.

### Setup
```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
poetry build && poetry install
```
### Test
```
poetry run pytest
```
### Prepare data
```
poetry run biggan.prepare
```
### Train
```
poetry run biggan.train
```
