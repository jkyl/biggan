import tempfile
import glob
import os

import tensorflow as tf

import biggan


def dummy_dataset():
    return tf.data.Dataset.from_tensor_slices((
        tf.random.uniform((8, 256, 256, 3)),
        tf.random.uniform((8, 4)))).batch(1, drop_remainder=True)


def test_build_and_train_model():
    model = biggan.build_model(
        channels=4,
        num_classes=4,
        latent_dim=4,
    )
    assert model.G.built
    assert model.D.built
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
