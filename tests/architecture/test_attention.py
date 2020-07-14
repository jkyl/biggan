import tensorflow as tf
import numpy as np

from biggan.architecture import Attention


def test_that_attention_models_long_range_dependencies():

    # Create an attention module.
    input_ = tf.keras.Input((64, 64, 8))
    model = tf.keras.Model(input_, Attention(input_))

    # Predict on totally zero input image.
    zeros_in = np.zeros((1, 64, 64, 8))
    zeros_out = model.predict(zeros_in)

    # Output should be everywhere zero.
    assert np.allclose(zeros_out, zeros_in)

    # Predict on delta-function input image.
    single_one_in = zeros_in.copy()
    single_one_in[0, 0, 0, 0] = 1.
    single_one_out = model.predict(single_one_in)

    # Output (delta) should be everywhere nonzero.
    assert not np.allclose(single_one_out, single_one_in)
