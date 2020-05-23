import tensorflow as tf
import argparse
import logging

from src.nets import Generator
from src.nets import Discriminator

from src.data import get_train_data
from src.data import get_strategy
from src.data import postprocess

from tensorflow.compat.v1.train import get_global_step
from tensorflow.compat.v1 import summary


def main(args=None):
  """
  Trains a BigGAN-deep on a preprocessed image dataset.
  """
  if args is None:
    args = parse_arguments()

  def hinge_loss(logits_real, logits_fake):
    """
    Represents the "hinge" GAN objective given
    discriminator outputs `logits_real` and `logits_fake`.

    Cf. https://arxiv.org/pdf/1802.05957.pdf,
    equations 16 and 17.
    """
    L_G = -tf.reduce_sum(logits_fake)
    L_D = tf.reduce_sum(tf.nn.relu(1. - logits_real))\
        + tf.reduce_sum(tf.nn.relu(1. + logits_fake))
    return [l * (1. / args.batch_size) for l in (L_G, L_D)]

  def minimize(loss, weights, optimizer):
    """
    Graph-mode minimization without gradient tape
    or eager function tracing, similar to TF-1.X.
    """
    return optimizer.apply_gradients(zip(
      optimizer.get_gradients(loss, weights), weights))

  def model_fn(features, labels, mode):
    """
    Constructs an EstimatorSpec encompassing the GAN
    training algorithm given some image `features`.
    """
    # Build the networks.
    G = Generator(args.channels, args.classes)
    D = Discriminator(args.channels, args.classes)
    G.summary(); D.summary()
     
    # Sample latent vector `z` from N(0, 1).
    z = tf.random.normal(dtype=tf.float32,
      shape=(features.shape[0], G.inputs[0].shape[-1]))

    # Make predictions.
    predictions = G([z, labels], training=True)
    logits_real = D([features, labels], training=True)
    logits_fake = D([predictions, labels], training=True)

    # Hinge loss function.
    L_G, L_D = hinge_loss(logits_real, logits_fake)

    # Dual Adam optimizers.
    G_adam = tf.optimizers.Adam(1e-4, 0., 0.999)
    D_adam = tf.optimizers.Adam(4e-4, 0., 0.999)

    # Group the generator and discriminator updates.
    train_op = tf.group(
      minimize(L_G, G.trainable_weights, G_adam),
      minimize(L_D, D.trainable_weights, D_adam),
      get_global_step().assign_add(1),
    )
    # Create some tensorboard summaries.
    summary.image('xhat', postprocess(predictions), 10)
    summary.image('x', postprocess(features), 10)
    summary.scalar('L_G', L_G)
    summary.scalar('L_D', L_D)

    # Return an EstimatorSpec.
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=L_D, train_op=train_op)

  # Enable log messages
  tf.get_logger().setLevel(logging.INFO)

  # Dispatch estimator training
  tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=args.model_dir,
    config=tf.estimator.RunConfig(
      train_distribute=None if args.debug
        else get_strategy(),
      save_checkpoints_secs=3600,
      save_summary_steps=100,
    )
  ).train(lambda:
    get_train_data(
      args.data_file,
      args.batch_size),
    steps=1000000 )

def parse_arguments():
  p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument(
    'data_file',
    type=str,
    help='.npy file containing preprocessed image data',
  )
  p.add_argument(
    'model_dir',
    type=str,
    help='directory in which to save checkpoints and summaries',
  )
  p.add_argument(
    '-bs',
    dest='batch_size',
    type=int,
    default=64,
    help='global (not per-replica) number of samples per update',
  )
  p.add_argument(
    '-ch',
    dest='channels',
    type=int,
    default=48,
    help='channel multiplier in G and D',
  )
  p.add_argument(
    '-cl',
    dest='classes',
    type=int,
    default=27,
    help='number of image classes',
  )
  p.add_argument(
    '--debug',
    action='store_true',
    help=argparse.SUPPRESS,
  )
  return p.parse_args()

if __name__ == '__main__':
  main()
