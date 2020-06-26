import tensorflow as tf
import numpy as np
import os

from typing import List, Tuple, Dict, Union

from .networks import Generator
from .networks import Discriminator

from .training import discriminator_hinge_loss
from .training import generator_hinge_loss
from .training import imperative_minimize

from .data import get_train_data
from .data import get_strategy
from .data import postprocess


class BigGAN(tf.keras.Model):
    """
    Implementation of 256x256x3 BigGAN in Keras.
    """
    def __init__(self, channels: int, num_classes: int):
        """
        Initializes the BigGAN model.
        """
        super().__init__()
        self.G = Generator(channels)
        self.D = Discriminator(channels)
        self.latent_dim = self.G.inputs[0].shape[-1]
        self.num_classes = num_classes

    def compile(
        self,
        G_learning_rate: Union[float, tf.Tensor] = 1e-4,
        D_learning_rate: Union[float, tf.Tensor] = 4e-4,
        global_batch_size: Union[int, None] = None,
    ):
        """
        Constructs dual optimizers for BigGAN training.
        """
        super().compile()
        self.G_adam = tf.optimizers.Adam(G_learning_rate, 0.0, 0.999)
        self.D_adam = tf.optimizers.Adam(D_learning_rate, 0.0, 0.999)

    def G_step(
        self,
        *,
        latent_z: tf.Tensor,
        labels: tf.Tensor,
    ) -> tf.Tensor:
        """
        Performs an update on the generator parameters, given
        a latent z-vector and a label index.
        """
        return imperative_minimize(
            optimizer=self.G_adam,
            loss_fn=lambda:
                generator_hinge_loss(
                    logits_fake=self.D([self.G([latent_z, labels]), labels]),
                    global_batch_size=self.global_batch_size,
                ),
            var_list=self.G.trainable_weights,
        )

    def D_step(
        self,
        *,
        features: tf.Tensor,
        latent_z: tf.Tensor,
        labels: tf.Tensor,
    ) -> tf.Tensor:
        """
        Performs an update on the discriminator parameters, given
        a batch of real images, a latent z-vector, and a label index,
        as well as a global batch size by which to scale the gradient
        signal.
        """
        return imperative_minimize(
            optimizer=self.D_adam,
            loss_fn=lambda:
                discriminator_hinge_loss(
                    logits_real=self.D([features, labels]),
                    logits_fake=self.D([self.G([latent_z, labels]), labels]),
                    global_batch_size=self.global_batch_size,
                ),
            var_list=self.D.trainable_weights,
        )

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        Performs a single training iteration on a batch of
        images and their class labels.
        """
        # Unpack data into images and class labels.
        features, labels = data

        # Get the (per-replica) batch size from the shape of the images.
        local_batch_size = features.shape[0]

        # Resolve the global batch size, if applicable.
        if self.global_batch_size is None:
            self.global_batch_size = local_batch_size

        # Sample a batch of latent vectors from the normal distribution.
        latent_z = tf.random.normal((local_batch_size, self.latent_dim))

        # Do a gradient descent update on the discriminator.
        L_D = self.D_step(
            features=features,
            latent_z=latent_z,
            labels=labels,
        )
        # Do a gradient descent update on the generator.
        L_G = self.G_step(
            latent_z=latent_z,
            labels=labels,
        )
        # Return both losses for logging.
        return {"L_G": L_G, "L_D": L_D}

    def create_callbacks(self, model_dir: str) -> List[tf.keras.callbacks.Callback]:
        """
        Creates a list of callbacks that handle model checkpointing and logging.
        """
        image_file_writer = tf.summary.create_file_writer(os.path.join(model_dir, "test"))
        def log_images(epoch, logs):
            z = np.random.normal(size=(10, self.latent_dim))
            c = np.random.randint(self.num_classes, size=10)
            xhat = self.G.predict([z, c])
            images = postprocess(xhat).numpy()
            with image_file_writer.as_default():
                for index, image in enumerate(images):
                    tf.summary.image(f"predictions/{index}", image[None], step=epoch)
            image_file_writer.flush()
        return [
            tf.keras.callbacks.LambdaCallback(on_epoch_end=log_images),
            tf.keras.callbacks.TensorBoard(log_dir=model_dir, write_graph=False),
            tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_dir, "ckpt_{epoch}")),
        ]
