import tempfile
import glob
import os

import pytest

import tensorflow as tf

import biggan

from biggan.config import base as cfg


@pytest.mark.parametrize("image_size", cfg.choices.image_size)
def test_build_and_train_model(image_size):
    model = biggan.build_model(
        image_size=image_size,
        channels=4,
        num_classes=4,
        latent_dim=4,
    )
    assert model.G.built
    assert model.D.built

    def dummy_dataset():
        return tf.data.Dataset.from_tensor_slices((
            tf.random.normal((2, image_size, image_size, 3)),
            tf.random.uniform((2, 4)))).batch(1, drop_remainder=True)

    with tempfile.TemporaryDirectory() as model_path:
        biggan.train_model(
            model=model,
            dataset=dummy_dataset(),
            model_path=model_path,
            num_epochs=1,
            log_every=1,
        )
        assert len(glob.glob(os.path.join(model_path, "ckpt_*"))) > 0
        assert len(glob.glob(os.path.join(model_path, "events.out.tfevents.*"))) > 0
