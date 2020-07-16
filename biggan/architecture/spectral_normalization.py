import tensorflow as tf

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
from tensorflow.keras import backend as K

from tensorflow.python.keras.utils import tf_utils

from typing import Union

from ..config import base as cfg


def spectrally_normalize_weight(
    weight: tf.Tensor,
    right_singular_vector: tf.Tensor,
    epsilon: Union[float, tf.Tensor] = cfg.defaults.epsilon,
    training: Union[None, bool, tf.Tensor] = None,
):
    """
    Given the kernel from a Conv2D or Dense layer,
    and an estimate of its right singular vector,
    performs spectral normalization on the kernel.
    """
    if training is None:
        training = K.learning_phase()

    def _l2normalize(v):
        return v / (K.sum(v ** 2) ** 0.5 + epsilon)

    def power_iteration(W, u):
        _u = u
        _v = _l2normalize(K.dot(_u, K.transpose(W)))
        _u = _l2normalize(K.dot(_v, W))
        return tf.stop_gradient(_u), tf.stop_gradient(_v)

    W_shape = weight.shape.as_list()
    W_reshaped = K.reshape(weight, [-1, W_shape[-1]])
    _u, _v = power_iteration(W_reshaped, right_singular_vector)
    sigma = K.dot(_v, W_reshaped)
    sigma = K.dot(sigma, K.transpose(_u))
    W_bar = W_reshaped / sigma

    def assign_update():
        with tf.control_dependencies([right_singular_vector.assign(_u)]):
            return K.reshape(W_bar, W_shape)

    W_bar = tf_utils.smart_cond(
        training, assign_update, lambda: K.reshape(W_bar, W_shape)
    )

    return W_bar


def create_right_singular_vector(layer: Layer):
    """
    Adds a non-trainable parameter to the layer that tracks
    estimates of the right singular vector of the layer's
    weight matrix.
    """
    return layer.add_weight(
        shape=tuple([1, layer.kernel.shape.as_list()[-1]]),
        initializer=initializers.RandomNormal(0, 1),
        name="sn",
        trainable=False,
        synchronization=tf.VariableSynchronization.ON_READ,
        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
    )


class SpectralConv2D(Conv2D):
    """
    Spectrally-normalized Conv2D layer.
    """
    def __init__(
        self,
        *args,
        epsilon=cfg.defaults.epsilon,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        """
        Builds the layer, including a non-trainable vector `u`
        which tracks estimates of the right singular vector of
        the weight matrix.
        """
        super().build(input_shape)
        self.u = create_right_singular_vector(self)

    def call(self, inputs, training=None):
        """
        Calls the layer on some input. If training,
        updates the `u` parameter with revised estimates
        of the right singular vector.
        """
        W_bar = spectrally_normalize_weight(
            weight=self.kernel,
            right_singular_vector=self.u,
            training=training,
            epsilon=self.epsilon,
        )
        outputs = K.conv2d(
            inputs,
            W_bar,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class SpectralDense(Dense):
    """
    Spectrally-normalized Dense layer.
    """

    def __init__(
        self,
        *args,
        epsilon=cfg.defaults.epsilon,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        """
        Builds the layer, including a non-trainable vector `u`
        which tracks estimates of the right singular vector of
        the weight matrix.
        """
        super().build(input_shape)
        self.u = create_right_singular_vector(self)

    def call(self, inputs, training=None):
        """
        Calls the layer on some input. If training,
        updates the `u` parameter with revised estimates
        of the right singular vector.
        """
        W_bar = spectrally_normalize_weight(
            weight=self.kernel,
            right_singular_vector=self.u,
            training=training,
            epsilon=self.epsilon,
        )
        output = K.dot(inputs, W_bar)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format="channels_last")
        if self.activation is not None:
            output = self.activation(output)
        return output
