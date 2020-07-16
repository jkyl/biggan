import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Dot
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras import Input
from functools import partial
from .spectral_normalization import SpectralConv2D
from .spectral_normalization import SpectralDense
from .attention import Attention
from ..config import base as cfg


# This dict defines the sequence of blocks, their channel multipliers,
# and their downsampling rules for each of 3 architectures.
block_sequences = {
    128: [
        {"ch": 2, "down": True},
        {"ch": 2},
        {"NonLocal"},
        {"ch": 4, "down": True},
        {"ch": 4},
        {"ch": 8, "down": True},
        {"ch": 8},
        {"ch": 16, "down": True},
        {"ch": 16},
        {"ch": 16, "down": True},
        {"ch": 16},
    ],
    256: [
        {"ch": 2, "down": True},
        {"ch": 2},
        {"ch": 4, "down": True},
        {"ch": 4},
        {"NonLocal"},
        {"ch": 8, "down": True},
        {"ch": 8},
        {"ch": 8, "down": True},
        {"ch": 8},
        {"ch": 16, "down": True},
        {"ch": 16},
        {"ch": 16, "down": True},
        {"ch": 16},
    ],
    512: [
        {"ch": 1, "down": True},
        {"ch": 1},
        {"ch": 2, "down": True},
        {"ch": 2},
        {"ch": 4, "down": True},
        {"ch": 4},
        {"NonLocal"},
        {"ch": 8, "down": True},
        {"ch": 8},
        {"ch": 8, "down": True},
        {"ch": 8},
        {"ch": 16, "down": True},
        {"ch": 16},
        {"ch": 16, "down": True},
        {"ch": 16},
    ],
}


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


def Discriminator(
    *,
    image_size: int,
    num_classes: int,
    ch: int = cfg.defaults.channels,
    epsilon: float = cfg.defaults.epsilon,
):
    """
    Cf. https://arxiv.org/pdf/1809.11096.pdf,
    table 8, right side.
    """

    # Hardcode some hyperparameters for all layers.
    SNConv2D = partial(SpectralConv2D, epsilon=epsilon, padding="same", kernel_initializer="orthogonal")
    SNDense = partial(SpectralDense, epsilon=epsilon, kernel_initializer="orthogonal")
    NonLocal = partial(Attention, epsilon=epsilon)

    def Block(x, output_dim, down=False):
        """
        Constructs a bottlenecked residual block
        with optional average pooling for biggan-deep's
        discriminator function, D.

        see https://arxiv.org/pdf/1809.11096.pdf,
        figure 16, right side.
        """
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

    # Input the real or fake image.
    image = x = Input((image_size, image_size, 3))

    # Project onto `ch` channels.
    x = SNConv2D(ch, 3)(x)

    # Apply the appropriate sequence of residual blocks.
    for block_config in block_sequences[image_size]:
        if "NonLocal" in block_config:
            x = NonLocal(x)
        else:
            output_dim = block_config["ch"] * ch
            down = block_config["down"] if "down" in block_config else False
            x = Block(x, output_dim=output_dim, down=down)

    # Perform global pooling.
    x = Activation("relu")(x)
    x = GlobalSumPooling2D()(x)

    # Input a one-hot class label and embed it.
    class_label = Input((num_classes,))
    embedding = SNDense(16 * ch, use_bias=False)(class_label)

    # Compute a conditional output by projecting the pooled output onto
    # the embedding, and adding it back to the regular logit.
    conditional_logit = Add()([SNDense(1)(x), Dot((1, 1))([x, embedding])])

    # Return a Keras model.
    return Model([image, class_label], conditional_logit, name="Discriminator")
