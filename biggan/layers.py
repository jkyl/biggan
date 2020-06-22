import tensorflow as tf

from tensorflow.keras import initializers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers.experimental import SyncBatchNormalization
from tensorflow.python.keras.utils import tf_utils

from typing import Union
from functools import wraps


__all__ = [
    "module",
    "TakeChannels",
    "GlobalSumPooling2D",
    "HyperBatchNorm",
    "SpectralConv2D",
    "SpectralDense",
]


def module(func):
    """
    Successive calls to `function` will be
    variable-scoped with non-conflicting names
    based on `function.__name__`
    """
    @wraps(func)
    def decorated(*args, **kwargs):
        with tf.compat.v1.variable_scope(None, default_name=func.__name__):
            return func(*args, **kwargs)

    return decorated


def TakeChannels(output_dim):
    """
    Layer that slices the first `output_dim`
    channels of a given tensor.
    """
    def call(x):
        return x[..., :output_dim]

    def output_shape(input_shape):
        return input_shape[:-1] + (output_dim,)

    return Lambda(call, output_shape=output_shape)


def GlobalSumPooling2D():
    """
    Layer that sums over all spatial locations,
    preserving batch and channels dimensions.
    """
    def call(x):
        return tf.reduce_sum(x, axis=(1, 2))

    def output_shape(input_shape):
        return input_shape[0], input_shape[-1]

    return Lambda(call, output_shape=output_shape)


@module
def HyperBatchNorm(x, z):
    """
    Builds and calls a cross-replica conditional
    batch normalization layer on an image tensor
    `x` and conditioning vector `z`.
    """
    dim = K.int_shape(x)[-1]
    if tf.distribute.get_replica_context() is None:
        BN = BatchNormalization
    else:
        BN = SyncBatchNormalization
    x = BN(scale=False, center=False)(x)
    gamma = Dense(dim, bias_initializer="ones")(z)
    beta = Dense(dim, bias_initializer="zeros")(z)
    return Lambda(
        lambda xgb: xgb[0] * xgb[1][:, None, None] + xgb[2][:, None, None],
        output_shape=lambda s: s[0],
    )([x, gamma, beta])


def spectrally_normalize_weight(
    weight: tf.Tensor,
    right_singular_vector: tf.Tensor,
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
        return v / (K.sum(v ** 2) ** 0.5 + 1e-6)

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
        W_bar = spectrally_normalize_weight(self.kernel, self.u, training=training)
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
        W_bar = spectrally_normalize_weight(self.kernel, self.u, training=training)
        output = K.dot(inputs, W_bar)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format="channels_last")
        if self.activation is not None:
            output = self.activation(output)
        return output
