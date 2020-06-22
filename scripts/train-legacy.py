import tensorflow as tf
import argparse
import logging

from biggan import Generator
from biggan import Discriminator

from biggan.data import get_train_data
from biggan.data import get_strategy
from biggan.data import postprocess

from biggan.training import declarative_minimize
from biggan.training import full_hinge_loss

from tensorflow.compat.v1.train import get_global_step
from tensorflow.compat.v1 import summary


def build_estimator_and_train(
    *,
    batch_size: int,
    channels: int,
    classes: int,
    model_dir: str,
    data_file: str,
    debug: bool = False,
):
    """
    Trains a BigGAN on a preprocessed image dataset.
    """

    def model_fn(features, labels, mode):
        """
        Constructs an EstimatorSpec encompassing the GAN
        training algorithm given some image `features`.
        """
        # Build the networks.
        G = Generator(channels, classes)
        D = Discriminator(channels, classes)
        G.summary()
        D.summary()

        # Sample latent vector `z` from N(0, 1).
        z = tf.random.normal(
            dtype=tf.float32, shape=(features.shape[0], G.inputs[0].shape[-1])
        )

        # Make predictions.
        predictions = G([z, labels], training=True)
        logits_real = D([features, labels], training=True)
        logits_fake = D([predictions, labels], training=True)

        # Hinge loss function.
        L_G, L_D = full_hinge_loss(logits_real=logits_real, logits_fake=logits_fake, global_batch_size=batch_size)

        # Dual Adam optimizers.
        G_adam = tf.optimizers.Adam(1e-4, 0.0, 0.999)
        D_adam = tf.optimizers.Adam(4e-4, 0.0, 0.999)

        # Group the generator and discriminator updates.
        train_op = tf.group(
            declarative_minimize(loss=L_G, var_list=G.trainable_weights, optimizer=G_adam),
            declarative_minimize(loss=L_D, var_list=D.trainable_weights, optimizer=D_adam),
            get_global_step().assign_add(1),
        )
        # Create some tensorboard summaries.
        summary.image("xhat", postprocess(predictions), 10)
        summary.image("x", postprocess(features), 10)
        summary.scalar("L_G", L_G)
        summary.scalar("L_D", L_D)

        # Return an EstimatorSpec.
        return tf.estimator.EstimatorSpec(mode=mode, loss=L_D, train_op=train_op)

    # Enable log messages
    tf.get_logger().setLevel(logging.INFO)

    # Dispatch estimator training
    tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        config=tf.estimator.RunConfig(
            train_distribute=None if debug else get_strategy(),
            save_checkpoints_secs=3600,
            save_summary_steps=100,
        ),
    ).train(lambda: get_train_data(data_file, batch_size), steps=1000000)


def parse_arguments():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "data_file",
        type=str,
        help=".npy file containing preprocessed image data",
    )
    p.add_argument(
        "model_dir",
        type=str,
        help="directory in which to save checkpoints and summaries",
    )
    p.add_argument(
        "-bs",
        dest="batch_size",
        type=int,
        default=64,
        help="global (not per-replica) number of samples per update",
    )
    p.add_argument(
        "-ch",
        dest="channels",
        type=int,
        default=48,
        help="channel multiplier in G and D",
    )
    p.add_argument(
        "-cl",
        dest="classes",
        type=int,
        default=27,
        help="number of image classes",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return p.parse_args()


if __name__ == "__main__":
    main()
