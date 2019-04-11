from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import argparse
import logging

from tensorflow.compat.v1.train import get_global_step
from tensorflow.compat.v1 import summary

from src.nets import Generator
from src.nets import Discriminator
from src.data import get_train_data
from src.data import get_strategy
from src.data import postprocess


def model_fn(features, mode, params):

  # set the learning phase and float precision
  tf.keras.backend.set_learning_phase(True)
  tf.keras.backend.set_floatx(tf.float16)
  features = tf.cast(features, tf.float16)

  # build the networks
  G = Generator(params['channels'])
  D = Discriminator(params['channels'])
  G.summary(); D.summary()
  
  # sample z from N(0, 1)
  z = tf.random.normal(dtype=tf.float16,
    shape=(tf.shape(features)[0], G.input_shape[-1]))

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

  # v1-style `minimize` (w/o gradient tape)
  def minimize(loss, weights, optimizer, iteration):
    with tf.control_dependencies([
        optimizer.apply_gradients(zip(
          optimizer.get_gradients(loss, weights), weights)),
        get_global_step().assign(G_adam.iterations)]):
      return tf.add(iteration, 1)

  # update D twice for every G update
  counter = tf.constant(0)
  train_op = tf.group(
    tf.while_loop(
      lambda i: tf.less(i, 2),
      lambda i: minimize(L_D, D.trainable_weights, D_adam, i), 
      [counter]), 
    minimize(L_G, G.trainable_weights, G_adam, counter))

  # create some tensorboard summaries
  summary.image('xhat', postprocess(predictions), 10)
  summary.image('x', postprocess(features), 10)
  summary.scalar('L_G', L_G)
  summary.scalar('L_D', L_D)
 
  # return an EstimatorSpec
  return tf.estimator.EstimatorSpec(
    mode=mode, loss=L_D, train_op=train_op)
 
def main(args):
  tf.get_logger().setLevel(logging.INFO)
  tf.estimator.Estimator(
    model_fn=model_fn,
    params=vars(args),
    model_dir=args.model_dir,
    config=tf.estimator.RunConfig(
      train_distribute=get_strategy(),
      save_checkpoints_secs=3600,
      save_summary_secs=60)
  ).train(get_train_data, steps=1000000)

if __name__ == '__main__':
  p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument('data_file', type=str,
    help='.npy file containing preprocessed image data')
  p.add_argument('model_dir', type=str,
    help='directory in which to save checkpoints and summaries')
  p.add_argument('-bs', dest='batch_size', type=int, default=64,
    help='number of samples per minibatch update')
  p.add_argument('-ch', dest='channels', type=int, default=16,
    help='channel multiplier in G and D')
  main(p.parse_args())
