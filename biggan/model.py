import warnings
import os

import tensorflow as tf

from typing import List, Tuple, Dict, Union, Callable
from contextlib import nullcontext

from .architecture import Generator, Discriminator
from .data import postprocess_image
from .config import base as cfg


class BigGAN(tf.keras.Model):
    """
    Implementation of BigGAN in Keras.
    """
    def __init__(
        self,
        *,
        image_size: int,
        channels: int = cfg.defaults.channels,
        latent_dim: int = cfg.defaults.latent_dim,
        momentum: float = cfg.defaults.momentum,
        epsilon: float = cfg.defaults.momentum,
        num_classes: int,
    ):
        """
        Initializes the BigGAN model.
        """
        super().__init__()
        self.G = Generator(
            image_size=image_size,
            ch=channels,
            num_classes=num_classes,
            latent_dim=latent_dim,
            momentum=momentum,
            epsilon=epsilon,
        )
        self.D = Discriminator(
            image_size=image_size,
            ch=channels,
            num_classes=num_classes,
            epsilon=epsilon,
        )
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.epsilon = epsilon

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

    def set_num_D_updates(self, num_D_updates: int):
        """
        Setter for num. D updates.
        """
        self._num_D_updates = num_D_updates

    @property
    def num_D_updates(self):
        """
        Returns the number of discriminator updates per generator update.
        """
        return self._num_D_updates

    def compile(
        self,
        G_learning_rate: float = cfg.defaults.G_learning_rate,
        D_learning_rate: float = cfg.defaults.D_learning_rate,
        G_beta_1: float = cfg.defaults.G_beta_1,
        D_beta_1: float = cfg.defaults.D_beta_1,
        G_beta_2: float = cfg.defaults.G_beta_2,
        D_beta_2: float = cfg.defaults.D_beta_2,
        num_D_updates: int = cfg.defaults.num_D_updates,
        global_batch_size: Union[int, None] = None,
    ):
        """
        Constructs dual optimizers for BigGAN training.
        """
        super().compile()
        self.G_adam = tf.optimizers.Adam(G_learning_rate, G_beta_1, G_beta_2, epsilon=self.epsilon)
        self.D_adam = tf.optimizers.Adam(D_learning_rate, D_beta_1, D_beta_2, epsilon=self.epsilon)
        self.set_num_D_updates(num_D_updates)
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

    def _do_train_step(
        self,
        *,
        features: tf.Tensor,
        labels: tf.Tensor,
        update_G: bool,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Implementation of the training step.
        """

        # Do the forward pass, recording gradients for the trainable parameters.
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(self.D.trainable_weights + self.G.trainable_weights)
            latent_vector = tf.nn.relu(tf.random.normal(shape=(labels.shape[0], self.latent_dim)))
            predictions = self.G([latent_vector, labels], training=True)
            logits_fake = self.D([predictions, labels], training=True)
            logits_real = self.D([features, labels], training=True)
            L_D = self.discriminator_hinge_loss(logits_fake=logits_fake, logits_real=logits_real)
            L_G = self.generator_hinge_loss(logits_fake=logits_fake)

        # Update the discriminator.
        self.D_adam.apply_gradients(
            zip(tape.gradient(L_D, self.D.trainable_weights), self.D.trainable_weights))

        # Update the generator.
        if update_G:
            self.G_adam.apply_gradients(
                zip(tape.gradient(L_G, self.G.trainable_weights), self.G.trainable_weights))

        # Return the outputs and losses for logging.
        return predictions, L_D, L_G

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
            warnings.warn("global batch size not set; using per-replica batch size.")
            self.set_global_batch_size(features.shape[0])

        # Determine whether to update the generator this step.
        update_G = self.D_adam.iterations % self.num_D_updates == 0

        # Do a training step and collect the predictions and losses.
        predictions, L_D, L_G = self._do_train_step(
            features=features,
            labels=labels,
            update_G=update_G,
        )
        # Write the images seen during training.
        return self.summarize(
            images={"features": features, "predictions": predictions},
            scalars={"L_D": L_D, "L_G": L_G})

    def summarize(self, *, scalars: Dict[str, tf.Tensor], images: Dict[str, tf.Tensor]):
        """
        Logs scalar and image tensors seen during training to the model path,
        and returns the scalars dict for progress bar reporting.
        """
        with self._summary_writer.as_default():
            step = self.D_adam.iterations - 1
            with tf.summary.record_if(step % self._log_every == 0):
                for key, scalar in scalars.items():
                    tf.summary.scalar(key, scalar, step=step)
                for key, image in images.items():
                    tf.summary.image(key, postprocess_image(image), step=step, max_outputs=16)
        return scalars

    def create_callbacks(
        self,
        model_path: str,
        *,
        log_every: int,
    ) -> List[tf.keras.callbacks.Callback]:
        """
        Creates a list of callbacks that handle model checkpointing and logging.
        """
        self._summary_writer = tf.summary.create_file_writer(model_path)
        self._log_every = log_every
        return [tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_path, "ckpt_{epoch}"))]


def get_strategy_scope(use_tpu=cfg.defaults.use_tpu):
    """
    Returns a mirrored strategy over all available GPUs,
    or falls back to CPU if no GPUs available.
    """
    if use_tpu:
        raise NotImplementedError("use_tpu=True")
    if len(tf.config.list_physical_devices("GPU")) > 1:
        return tf.distribute.MirroredStrategy().scope()
    return nullcontext()


def build_model(
    *,
    image_size: int,
    channels: int = cfg.defaults.channels,
    num_classes: Union[int, Callable], # Determined lazily from the dataset.
    latent_dim: int = cfg.defaults.latent_dim,
    checkpoint: str = None,
    G_learning_rate: float = cfg.defaults.G_learning_rate,
    D_learning_rate: float = cfg.defaults.D_learning_rate,
    G_beta_1: float = cfg.defaults.G_beta_1,
    D_beta_1: float = cfg.defaults.D_beta_1,
    G_beta_2: float = cfg.defaults.G_beta_2,
    D_beta_2: float = cfg.defaults.D_beta_2,
    num_D_updates: int = cfg.defaults.num_D_updates,
    global_batch_size: Union[int, None] = None,
    use_tpu: bool = cfg.defaults.use_tpu,
    momentum: float = cfg.defaults.momentum,
    epsilon: float = cfg.defaults.epsilon,
):
    """
    Builds the model within a distribution strategy context.
    """
    with get_strategy_scope(use_tpu=use_tpu):
        model = BigGAN(
            image_size=image_size,
            channels=channels,
            num_classes=(num_classes() if callable(num_classes) else num_classes),
            latent_dim=latent_dim,
            momentum=momentum,
            epsilon=epsilon,
        )
        model.compile(
            G_learning_rate=G_learning_rate,
            D_learning_rate=D_learning_rate,
            G_beta_1=G_beta_1,
            D_beta_1=D_beta_1,
            G_beta_2=G_beta_2,
            D_beta_2=D_beta_2,
            num_D_updates=num_D_updates,
            global_batch_size=global_batch_size,
        )
        if checkpoint is not None:
            model.load_weights(checkpoint)
    return model


def train_model(
    *,
    model: BigGAN,
    dataset: tf.data.Dataset,
    model_path: str,
    num_epochs: int = cfg.defaults.num_epochs,
    log_every: int = cfg.defaults.log_every,
    initial_epoch: int = 0,
):
    """
    Train the built model on a dataset.
    """
    # Create the saving and logging callbacks.
    callbacks = model.create_callbacks(model_path, log_every=log_every)

    # Set the global batch size for distributed training.
    model.set_global_batch_size(dataset.element_spec[0].shape[0])

    # Fit the model to the data, calling the callbacks every `log_every` steps.
    model.fit(
        dataset,
        callbacks=callbacks,
        epochs=num_epochs,
        initial_epoch=initial_epoch,
    )
    # Return the trained model.
    return model
