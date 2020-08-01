import tempfile
import glob
import os

import tensorflow as tf
import pytest
import biggan

from biggan.config import base as cfg
from itertools import product as outer
from tensorflow.keras.mixed_precision import experimental as mp


@pytest.mark.parametrize(
    "image_size,mixed_precision",
    list(outer(cfg.choices.image_size, [True, False]))
)
def test_build_and_train_model(image_size, mixed_precision):

    if mixed_precision:
        mp.set_policy(mp.Policy("mixed_float16"))

    model = biggan.build_model(
        image_size=image_size,
        channels=4,
        num_classes=4,
        latent_dim=4,
    )
    assert model.G.built and model.D.built

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
