import tensorflow as tf
import numpy as np

from biggan.architecture.conditional_batch_normalization import ConditionalBatchNormalization


# Create a conditional batch normalization module.
x = tf.keras.Input((64, 64, 8))
z = tf.keras.Input((8,))
model = tf.keras.Model([x, z], ConditionalBatchNormalization(x, z))


def test_that_it_regresses_to_standard_batchnorm():

    # Run CBN on a zero vector input.
    image = np.random.uniform(size=((1,) + x.shape[1:]))
    vector = np.zeros((1,) + z.shape[1:])
    output = model([image, vector], training=True).numpy()

    # The outputs should have zero mean and unit std.
    assert np.allclose(output.mean(axis=(0, 1, 2)), 0., atol=1e-4)
    assert np.allclose(output.std(axis=(0, 1, 2)), 1., atol=1e-4)


def test_that_it_is_conditioned_by_the_vector_input():

    # Run CBN on a dense vector input.
    image = np.random.uniform(size=((1,) + x.shape[1:]))
    vector = np.random.uniform(size=((1,) + z.shape[1:]))
    output = model([image, vector], training=True).numpy()

    # The outputs should *not* have zero mean and unit std.
    assert not np.allclose(output.mean(axis=(0, 1, 2)), 0., atol=1e-4)
    assert not np.allclose(output.std(axis=(0, 1, 2)), 1., atol=1e-4)


def test_that_it_has_the_correct_number_and_type_of_weights():

    # CBN should have two trainable Dense layers with biases.
    assert len(model.trainable_weights) == 4

    # Get the names of those variables.
    weight_types = [
        w.name.split("/")[-1].split(":")[0]
        for w in model.trainable_weights]

    # Check that the variables are two kernels and two biases.
    assert weight_types == ["kernel", "bias"] * 2
