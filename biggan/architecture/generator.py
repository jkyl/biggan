import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Add
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from functools import partial
from .conditional_batch_normalization import ConditionalBatchNormalization
from .spectral_normalization import SpectralConv2D
from .spectral_normalization import SpectralDense
from .attention import Attention
from ..config import base as cfg


# This dict defines the sequence of blocks, their channel multipliers,
# and their downsampling rules for each of 3 architectures.
block_sequences = {
    128: [
        {"ch": 16},
        {"ch": 16, "up": True},
        {"ch": 16},
        {"ch": 8, "up": True},
        {"ch": 8},
        {"ch": 4, "up": True},
        {"ch": 4},
        {"ch": 2, "up": True},
        {"NonLocal"},
        {"ch": 2},
        {"ch": 1, "up": True},
    ],
    256: [
        {"ch": 16},
        {"ch": 16, "up": True},
        {"ch": 16},
        {"ch": 8, "up": True},
        {"ch": 8},
        {"ch": 8, "up": True},
        {"ch": 8},
        {"ch": 4, "up": True},
        {"NonLocal"},
        {"ch": 4},
        {"ch": 2, "up": True},
        {"ch": 2},
        {"ch": 1, "up": True},
    ],
    512: [
        {"ch": 16},
        {"ch": 16, "up": True},
        {"ch": 16},
        {"ch": 8, "up": True},
        {"ch": 8},
        {"ch": 8, "up": True},
        {"ch": 8},
        {"ch": 4, "up": True},
        {"NonLocal"},
        {"ch": 4},
        {"ch": 2, "up": True},
        {"ch": 2},
        {"ch": 1, "up": True},
        {"ch": 1},
        {"ch": 1, "up": True},
    ],
}


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


def Generator(
    *,
    image_size: int,
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

    # Use the same momentum and epsilon in all batch/spectral norm layers.
    BatchNorm = partial(ConditionalBatchNormalization, momentum=momentum, epsilon=epsilon)
    SNConv2D = partial(SpectralConv2D, epsilon=epsilon, padding="same", kernel_initializer="orthogonal")
    SNDense = partial(SpectralDense, epsilon=epsilon, kernel_initializer="orthogonal")
    NonLocal = partial(Attention, epsilon=epsilon)

    def Block(
        x: tf.Tensor,
        z: tf.Tensor,
        output_dim: int,
        *,
        up: bool = False
    ):
        """
        Constructs and calls a bottlenecked residual block
        with conditional batch normalization and optional
        upsampling for biggan-deep's generator function.

        see https://arxiv.org/pdf/1809.11096.pdf,
        figure 16, left side.
        """
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

    # Input z-vector.
    z = Input((latent_dim,))

    # Input class label.
    y = Input((num_classes,))

    # Class embedding.
    y_emb = SNDense(latent_dim, use_bias=False)(y)

    # Concatenate with z.
    c = Concatenate()([z, y_emb])

    # Project and reshape.
    x = SNDense(4 * 4 * 16 * ch, use_bias=False)(c)
    x = Reshape((4, 4, 16 * ch))(x)

    # Apply the appropriate sequence of residual blocks.
    for block_config in block_sequences[image_size]:
        if "NonLocal" in block_config:
            x = NonLocal(x)
        else:
            output_dim = block_config["ch"] * ch
            up = block_config["up"] if "up" in block_config else False
            x = Block(x, c, output_dim=output_dim, up=up)

    # (256, 256, 1ch) -> (256, 256, 3)
    x = BatchNorm(x, c)
    x = Activation("relu")(x)
    x = SNConv2D(3, 3)(x)
    x = Activation("tanh")(x)

    # Return Keras model.
    return Model([z, y], x, name="Generator")
