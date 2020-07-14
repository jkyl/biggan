# Install from source
BigGAN uses [Poetry](https://python-poetry.org) as a build manager. Please refer to 
[their documentation](https://python-poetry.org/docs/) for detailed descriptions of 
these commands.

### Get Poetry
```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
```
### Build BigGAN
```
poetry build
```
### Install BigGAN
```
poetry install
```
### Run tests
```
poetry run pytest
```
