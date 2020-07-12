import tensorflow as tf

from tensorflow.keras.layers.experimental import SyncBatchNormalization
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Dot
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.python.keras.utils import tf_utils

from functools import partial

from typing import Union

from .config import base as cfg


def TakeChannels(output_dim: int):
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


def ConditionalBatchNormalization(
    x: tf.Tensor,
    z: tf.Tensor,
    *,
    epsilon: Union[float, tf.Tensor] = cfg.defaults.epsilon,
    momentum: Union[float, tf.Tensor] = cfg.defaults.momentum,
):
    """
    Builds and calls a cross-replica conditional
    batch normalization layer on an image tensor
    `x` and conditioning vector `z`.
    """

    if tf.distribute.in_cross_replica_context():
        BatchNorm = SyncBatchNormalization
    else:
        BatchNorm = BatchNormalization

    gamma = Dense(x.shape[-1], bias_initializer="ones")(z)
    beta = Dense(x.shape[-1], bias_initializer="zeros")(z)
    x = BatchNorm(scale=False, center=False, epsilon=epsilon, momentum=momentum)(x)

    def call(args):
        x, g, b = args
        return x * g[:, None, None] + b[:, None, None]

    def output_shape(input_shapes):
        return input_shapes[0]

    return Lambda(call, output_shape=output_shape)([x, gamma, beta])


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
        padding="same",
        kernel_initializer="orthogonal",
        epsilon=cfg.defaults.epsilon,
        **kwargs,
    ):
        super().__init__(
            *args,
            padding=padding,
            kernel_initializer=kernel_initializer,
            **kwargs,
        )
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
        kernel_initializer="orthogonal",
        epsilon=cfg.defaults.epsilon,
        **kwargs,
    ):
        super().__init__(
            *args,
            kernel_initializer=kernel_initializer,
            **kwargs,
        )
        self.epsilon=epsilon

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


def GBlock(
    x: tf.Tensor,
    z: tf.Tensor,
    output_dim: int,
    *,
    momentum: float = cfg.defaults.momentum,
    epsilon: float = cfg.defaults.epsilon,
    up: bool = False
):
    """
    Constructs and calls a bottlenecked residual block
    with conditional batch normalization and optional
    upsampling for biggan-deep's generator function.

    see https://arxiv.org/pdf/1809.11096.pdf,
    figure 16, left side.
    """
    BatchNorm = partial(ConditionalBatchNormalization, momentum=momentum, epsilon=epsilon)
    SNConv2D = partial(SpectralConv2D, epsilon=epsilon)
    input_dim = K.int_shape(x)[-1]
    x0 = x
    x = BatchNorm(x, z)
    x = Activation("relu")(x)
    x = SNConv2D(input_dim // 4, 1, use_bias=False)(x)
    x = BatchNorm(x, z)
    x = Activation("relu")(x)
    if up:
        x = UpSampling2D()(x)
    x = SNConv2D(input_dim // 4, 3, use_bias=False)(x)
    x = BatchNorm(x, z)
    x = Activation("relu")(x)
    x = SNConv2D(input_dim // 4, 3, use_bias=False)(x)
    x = BatchNorm(x, z)
    x = Activation("relu")(x)
    x = SNConv2D(output_dim, 1, use_bias=False)(x)
    if input_dim > output_dim:
        x0 = TakeChannels(output_dim)(x0)
    elif input_dim < output_dim:
        raise ValueError
    if up:
        x0 = UpSampling2D()(x0)
    return Add()([x, x0])


def DBlock(x, output_dim, down=False, epsilon=cfg.defaults.epsilon):
    """
    Constructs a bottlenecked residual block
    with optional average pooling for biggan-deep's
    discriminator function, D.

    see https://arxiv.org/pdf/1809.11096.pdf,
    figure 16, right side.
    """
    SNConv2D = partial(SpectralConv2D, epsilon=epsilon)
    input_dim = K.int_shape(x)[-1]
    x0 = x
    x = Activation("relu")(x)
    x = SNConv2D(output_dim // 4, 1)(x)
    x = Activation("relu")(x)
    x = SNConv2D(output_dim // 4, 3)(x)
    x = Activation("relu")(x)
    x = SNConv2D(output_dim // 4, 3)(x)
    x = Activation("relu")(x)
    if down:
        x = AveragePooling2D()(x)
        x0 = AveragePooling2D()(x0)
    if input_dim < output_dim:
        extra = output_dim - input_dim
        x0_extra = SNConv2D(extra, 1, use_bias=False)(x0)
        x0 = Concatenate()([x0, x0_extra])
    elif input_dim > output_dim:
        raise ValueError
    x = SNConv2D(output_dim, 1)(x)
    return Add()([x, x0])


def Attention(x, use_bias=True, epsilon=cfg.defaults.epsilon):
    """
    Constructs a self-attention layer.
    Cf. https://arxiv.org/pdf/1805.08318.pdf,
    section 3; also see the corresponding code:
    https://github.com/brain-research/self-attention-gan
    """
    SNConv2D = partial(SpectralConv2D, epsilon=epsilon)
    batch, height, width, channels = K.int_shape(x)
    space = height * width
    f = SNConv2D(channels // 8, 1, use_bias=False)(x)
    f = Reshape((space, channels // 8))(f)
    xbar = AveragePooling2D()(x)
    g = SNConv2D(channels // 8, 1, use_bias=False)(xbar)
    g = Reshape((space // 4, channels // 8))(g)
    h = SNConv2D(channels // 2, 1, use_bias=False)(xbar)
    h = Reshape((space // 4, channels // 2))(h)
    attn = Dot((2, 2))([f, g])
    attn = Activation("softmax")(attn)
    y = Dot((2, 1))([attn, h])
    y = Reshape((height, width, channels // 2))(y)
    y = SNConv2D(channels, 1, use_bias=use_bias)(y)
    return Add()([x, y])


def Generator(
    *,
    num_classes: int,
    ch: int = cfg.defaults.channels,
    latent_dim: int = cfg.defaults.latent_dim,
    momentum: float = cfg.defaults.momentum,
    epsilon: float = cfg.defaults.epsilon,
):
    """
    Cf. https://arxiv.org/pdf/1809.11096.pdf,
    table 8, left side.
    """

    # Use the same momentum in all batch norm layers.
    Block = partial(GBlock, momentum=momentum, epsilon=epsilon)

    # Input z-vector.
    z = Input((latent_dim,))

    # Input class label.
    y = Input((num_classes,))

    # Class embedding.
    y_emb = SpectralDense(latent_dim, use_bias=False, epsilon=epsilon)(y)

    # Concatenate with z.
    c = Concatenate()([z, y_emb])

    # Project and reshape.
    x = SpectralDense(4 * 4 * 16 * ch, use_bias=False, epsilon=epsilon)(c)
    x = Reshape((4, 4, 16 * ch))(x)

    # (4, 4, 16ch) -> (8, 8, 16ch)
    x = Block(x, c, 16 * ch)
    x = Block(x, c, 16 * ch, up=True)

    # (8, 8, 16ch) -> (16, 16, 8ch)
    x = Block(x, c, 16 * ch)
    x = Block(x, c, 8 * ch, up=True)

    # (16, 16, 8ch) -> (32, 32, 8ch)
    x = Block(x, c, 8 * ch)
    x = Block(x, c, 8 * ch, up=True)

    # (32, 32, 8ch) -> (64, 64, 4ch)
    x = Block(x, c, 8 * ch)
    x = Block(x, c, 4 * ch, up=True)

    # Non-local @ (64, 64, 4ch)
    x = Attention(x, use_bias=False, epsilon=epsilon)

    # (64, 64, 4ch) -> (128, 128, 2ch)
    x = Block(x, c, 4 * ch)
    x = Block(x, c, 2 * ch, up=True)

    # (128, 128, 2ch) -> (256, 256, 1ch)
    x = Block(x, c, 2 * ch)
    x = Block(x, c, 1 * ch, up=True)

    # (256, 256, 1ch) -> (256, 256, 3)
    x = ConditionalBatchNormalization(x, c, momentum=momentum, epsilon=epsilon)
    x = Activation("relu")(x)
    x = SpectralConv2D(3, 3, epsilon=epsilon)(x)
    x = Activation("tanh")(x)

    # Return Keras model.
    return Model([z, y], x, name="Generator")


def Discriminator(
    *,
    num_classes: int,
    ch: int = cfg.defaults.channels,
    epsilon: float = cfg.defaults.epsilon,
):
    """
    Cf. https://arxiv.org/pdf/1809.11096.pdf,
    table 8, right side.
    """
    # Use the same epsilon value in all residual blocks.
    Block = partial(DBlock, epsilon=epsilon)

    # Class embedding for projection discriminator.
    y = Input((num_classes,))
    y_emb = SpectralDense(16 * ch, use_bias=False, epsilon=epsilon)(y)

    # (256, 256, 3) -> (256, 256, 1ch)
    x = inp = Input((256, 256, 3))
    x = SpectralConv2D(ch, 3, epsilon=epsilon)(x)

    # (256, 256, 1ch) -> (128, 128, 2ch)
    x = Block(x, 2 * ch, down=True)
    x = Block(x, 2 * ch)

    # (128, 128, 2ch) -> (64, 64, 4ch)
    x = Block(x, 4 * ch, down=True)
    x = Block(x, 4 * ch)

    # Non-local @ (64, 64, 4ch)
    x = Attention(x, epsilon=epsilon)

    # (64, 64, 4ch) -> (32, 32, 8ch)
    x = Block(x, 8 * ch, down=True)
    x = Block(x, 8 * ch)

    # (32, 32, 8ch) -> (16, 16, 8ch)
    x = Block(x, 8 * ch, down=True)
    x = Block(x, 8 * ch)

    # (16, 16, 8ch) -> (8, 8, 16ch)
    x = Block(x, 16 * ch, down=True)
    x = Block(x, 16 * ch)

    # (8, 8, 16ch) -> (4, 4, 16ch)
    x = Block(x, 16 * ch, down=True)
    x = Block(x, 16 * ch)

    # (4, 4, 16ch) -> (16ch,)
    x = Activation("relu")(x)
    x = GlobalSumPooling2D()(x)

    # Conditional logit.
    x = Add()([SpectralDense(1, epsilon=epsilon)(x), Dot((1, 1))([x, y_emb])])

    # Return Keras model.
    return Model([inp, y], x, name="Discriminator")
