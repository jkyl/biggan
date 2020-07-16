import tensorflow as tf
import numpy as np

from biggan.architecture.attention import Attention


# Create an attention module.
inp = tf.keras.Input((64, 64, 8))
model = tf.keras.Model(inp, Attention(inp))


def test_that_it_captures_long_range_dependencies():

    # Predict on totally zero input image.
    zeros_in = np.zeros((1,) + inp.shape[1:])
    zeros_out = model.predict(zeros_in)

    # Output residual should be everywhere zero.
    assert np.allclose(zeros_out, zeros_in)

    # Predict on delta-function input image.
    single_one_in = zeros_in.copy()
    single_one_in[0, 0, 0, 0] = 1.
    single_one_out = model.predict(single_one_in)

    # Output residual should be everywhere nonzero.
    assert not np.allclose(single_one_out, single_one_in)


def test_that_it_has_the_correct_number_and_type_of_trainable_weights():

    # Attention should have 4 convolution layers, and the last one should
    # have a bias vector.
    assert len(model.trainable_weights) == 5

    # Get the names of those variables.
    weight_types = [
        w.name.split("/")[-1].split(":")[0]
        for w in model.trainable_weights]

    # Check that the first 4 are kernels.
    assert all([wt == "kernel" for wt in weight_types[:4]])

    # Check that the last one is a bias.
    assert weight_types[-1] == "bias"
