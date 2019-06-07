from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

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


def main(args):
  '''Trains a BigGAN-deep on some preprocessed images
  '''
  def model_fn(features, mode):
    '''Constructs an EstimatorSpec encompassing the GAN
    training algorithm on some `features` tensor
    '''
    # build the networks
    G = Generator(args.channels)
    D = Discriminator(args.channels)
    G.summary(); D.summary()

    # sample latent vector `z` from N(0, 1)
    z = tf.random.normal(dtype=tf.float32,
      shape=(features.shape[0], G.input_shape[-1]))

    # make predictions
    predictions = G(z)
    logits_real = D(features)
    logits_fake = D(predictions)

    # hinge loss function
    L_G = -tf.reduce_mean(logits_fake)
    L_D = tf.reduce_mean(tf.nn.relu(1. - logits_real))\
        + tf.reduce_mean(tf.nn.relu(1. + logits_fake))

    # dual Adam optimizers
    G_adam = tf.optimizers.Adam(1e-4, 0., 0.999, 1e-4)
    D_adam = tf.optimizers.Adam(4e-4, 0., 0.999, 1e-4)

    # graph-mode `minimize`
    def minimize(loss, weights, optimizer):
      return optimizer.apply_gradients(zip(
        optimizer.get_gradients(loss, weights), weights))

    # group the generator and discriminator updates
    train_op = tf.group(
      minimize(L_G, G.trainable_weights, G_adam),
      minimize(L_D, D.trainable_weights, D_adam),
      get_global_step().assign_add(1),
    )
    # create some tensorboard summaries
    summary.image('xhat', postprocess(predictions), 10)
    summary.image('x', postprocess(features), 10)
    summary.scalar('L_G', L_G)
    summary.scalar('L_D', L_D)

    # return an EstimatorSpec
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=L_D, train_op=train_op)

  # enable log messages
  tf.get_logger().setLevel(logging.INFO)

  # dispatch estimator training
  tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=args.model_dir,
    config=tf.estimator.RunConfig(
      train_distribute=get_strategy(),
      save_checkpoints_secs=3600,
      save_summary_steps=100,
    )
  ).train(lambda:
    get_train_data(
      args.data_file,
      args.batch_size),
    steps=1000000,
  )

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
    default=32,
    help='channel multiplier in G and D',
  )
  return p.parse_args()

if __name__ == '__main__':
  main(parse_arguments())
