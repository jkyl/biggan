import warnings
import os

import tensorflow as tf
import numpy as np

from typing import List, Tuple, Dict, Union

from .architecture import Generator, Discriminator
from .data import postprocess_image
from .config import base as cfg


class BigGAN(tf.keras.Model):
    """
    Implementation of 256x256x3 BigGAN in Keras.
    """
    def __init__(
        self,
        *,
        channels: int = cfg.defaults.channels,
        latent_dim: int = cfg.defaults.latent_dim,
        num_classes: int,
    ):
        """
        Initializes the BigGAN model.
        """
        super().__init__()
        self.G = Generator(
            ch=channels,
            num_classes=num_classes,
            latent_dim=latent_dim,
        )
        self.D = Discriminator(
            ch=channels,
            num_classes=num_classes,
        )
        self.num_classes = num_classes
        self.latent_dim = latent_dim

    def set_global_batch_size(self, global_batch_size: int):
        """
        Setter for global batch size.
        """
        self._global_batch_size = global_batch_size

    @property
    def global_batch_size(self):
        """
        Returns the total number of samples used to compute a training step.
        """
        return self._global_batch_size

    def compile(
        self,
        G_learning_rate: float = cfg.defaults.G_learning_rate,
        D_learning_rate: float = cfg.defaults.D_learning_rate,
        G_beta_1: float = cfg.defaults.G_beta_1,
        D_beta_1: float = cfg.defaults.D_beta_1,
        G_beta_2: float = cfg.defaults.G_beta_2,
        D_beta_2: float = cfg.defaults.D_beta_2,
        global_batch_size: Union[int, None] = None,
    ):
        """
        Constructs dual optimizers for BigGAN training.
        """
        super().compile()
        self.G_adam = tf.optimizers.Adam(G_learning_rate, G_beta_1, G_beta_2)
        self.D_adam = tf.optimizers.Adam(D_learning_rate, D_beta_1, D_beta_2)
        self.set_global_batch_size(global_batch_size)

    def discriminator_hinge_loss(
        self,
        *,
        logits_real: tf.Tensor,
        logits_fake: tf.Tensor,
    ) -> tf.Tensor:
        """
        Computes the "hinge" discriminator loss given two
        discriminator outputs, `logits_real` and `logits_fake`.

        Cf. Miyato, https://arxiv.org/pdf/1802.05957.pdf,
        equation 16.
        """
        L_D = tf.reduce_sum(tf.nn.relu(1.0 - logits_real)) \
            + tf.reduce_sum(tf.nn.relu(1.0 + logits_fake))
        return L_D * (1.0 / self.global_batch_size)

    def generator_hinge_loss(
        self,
        *,
        logits_fake: tf.Tensor,
    ) -> tf.Tensor:
        """
        Computes the "hinge" generator loss given one
        discriminator output, `logits_fake`.

        Cf. Miyato, https://arxiv.org/pdf/1802.05957.pdf,
        equation 17.
        """
        L_G = -tf.reduce_sum(logits_fake)
        return L_G * (1.0 / self.global_batch_size)

    def _do_train_step(self, *, features: tf.Tensor, labels: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Implementation of the training step.
        """
        def forward_pass():
            """
            Executes a complete forward pass on both generator and discriminator,
            returning their respective losses.
            """
            predictions = self.G([tf.random.normal(shape=(labels.shape[0], self.latent_dim)), labels], training=True)
            logits_fake = self.D([predictions, labels], training=True)
            logits_real = self.D([features, labels], training=True)
            L_D = self.discriminator_hinge_loss(logits_fake=logits_fake, logits_real=logits_real)
            L_G = self.generator_hinge_loss(logits_fake=logits_fake)
            return L_D, L_G

        # Do the forward pass, recording gradients for the trainable parameters.
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(self.G.trainable_weights + self.D.trainable_weights)
            L_D, L_G = forward_pass()

        # Apply the Adam optimizers on the parameters + their gradients.
        self.D_adam.apply_gradients(
            zip(tape.gradient(L_D, self.D.trainable_weights), self.D.trainable_weights))
        self.G_adam.apply_gradients(
            zip(tape.gradient(L_G, self.G.trainable_weights), self.G.trainable_weights))

        # Return the losses for logging.
        return {"L_D": L_D, "L_G": L_G}

    @tf.function
    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        Performs a single training iteration on a batch of
        images and their class labels.
        """
        # Unpack data into images and class labels.
        features, labels = data

        # Resolve the global batch size, if applicable.
        if self.global_batch_size is None:
            warnings.warn("`global_batch_size` not set; using per-replica batch size.")
            self.set_global_batch_size(features.shape[0])

        # Do a training step and return the losses for logging.
        return self._do_train_step(features=features, labels=labels)

    def create_callbacks(self, model_path: str, log_every: int) -> List[tf.keras.callbacks.Callback]:
        """
        Creates a list of callbacks that handle model checkpointing and logging.
        """
        image_file_writer = tf.summary.create_file_writer(os.path.join(model_path, "test"))
        def log_images(*args, **kwargs):
            step = self.G_adam.iterations.numpy() - 1
            if step % log_every != 0:
                return
            z = np.tile(np.random.normal(size=self.latent_dim), (self.num_classes, 1))
            c = np.eye(self.num_classes, dtype=np.float32)
            xhat = self.G([z, c], training=False)
            images = postprocess_image(xhat).numpy()
            with image_file_writer.as_default():
                for index, image in enumerate(images):
                    tf.summary.image(f"predictions/{index}", image[None], step=step)
        return [
            tf.keras.callbacks.LambdaCallback(on_batch_end=log_images),
            tf.keras.callbacks.TensorBoard(log_dir=model_path, write_graph=False, update_freq=log_every),
            tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_path, "ckpt_{epoch}")),
        ]


def get_strategy(use_tpu=cfg.defaults.use_tpu):
    """
    Returns a mirrored strategy over all available GPUs,
    or falls back to CPU if no GPUs available
    """
    if use_tpu:
        raise NotImplementedError("use_tpu=True")
    return tf.distribute.MirroredStrategy(devices=(
        [d.name for d in tf.config.list_physical_devices("GPU")] or ["/CPU:0"]))


def build_model(
    *,
    channels: int = cfg.defaults.channels,
    num_classes: Union[int, callable], # Determined lazily from the dataset.
    latent_dim: int = cfg.defaults.latent_dim,
    checkpoint: str = None,
    G_learning_rate: float = cfg.defaults.G_learning_rate,
    D_learning_rate: float = cfg.defaults.D_learning_rate,
    G_beta_1: float = cfg.defaults.G_beta_1,
    D_beta_1: float = cfg.defaults.D_beta_1,
    G_beta_2: float = cfg.defaults.G_beta_2,
    D_beta_2: float = cfg.defaults.D_beta_2,
    global_batch_size: Union[int, None] = None,
    use_tpu: bool = cfg.defaults.use_tpu,
):
    """
    Builds the model within a distribution strategy context.
    """
    with get_strategy(use_tpu=use_tpu).scope():
        model = BigGAN(
            channels=channels,
            num_classes=(num_classes() if callable(num_classes) else num_classes),
            latent_dim=latent_dim,
        )
        model.compile(
            G_learning_rate=G_learning_rate,
            D_learning_rate=D_learning_rate,
            G_beta_1=G_beta_1,
            D_beta_1=D_beta_1,
            G_beta_2=G_beta_2,
            D_beta_2=D_beta_2,
            global_batch_size=global_batch_size,
        )
        if checkpoint is not None:
            model.load_weights(checkpoint)
    return model


def train_model(
    *,
    model: BigGAN,
    data: tf.data.Dataset,
    model_path: str,
    num_epochs: int = cfg.defaults.num_epochs,
    log_every: int = cfg.defaults.log_every,
):
    """
    Train the built model on a dataset.
    """
    # Create the saving and logging callbacks.
    callbacks = model.create_callbacks(model_path, log_every=log_every)

    # Set the global batch size for distributed training.
    model.set_global_batch_size(data.element_spec[0].shape[0])

    # Fit the model to the data, calling the callbacks every `log_every` steps.
    model.fit(data, callbacks=callbacks, epochs=num_epochs)

    # Return the trained model.
    return model
