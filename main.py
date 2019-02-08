from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import argparse
import sys

from src import nets, data

def model_fn(features, labels, mode, params):
  del labels # unused

  # set the learning phase and float precision
  tf.keras.backend.set_learning_phase(True)
  tf.keras.backend.set_floatx(params['dtype'])
  features = tf.cast(features, params['dtype'])

  # build the networks
  G = nets.Generator(params['channels'])
  D = nets.Discriminator(params['channels'])

  # sample z from max(N(0,1), 0)
  z = tf.random_normal((
    params['batch_size'] // len(data.get_gpus()),
    params['z_dim']), dtype=params['dtype'])
  z = tf.maximum(z, tf.zeros_like(z))

  # make predictions
  predictions = G(z)
  logits_real = D(features)
  logits_fake = D(predictions)

  for l in G.layers:
    print(l)
    print(l.trainable_weights)

  # hinge loss function
  L_G = -tf.reduce_mean(logits_fake)
  L_D = tf.reduce_mean(tf.nn.relu(1. - logits_real))\
      + tf.reduce_mean(tf.nn.relu(1. + logits_fake))

  # two-timescale update rule
  G_opt = tf.train.AdamOptimizer(1e-4, 0., 0.999, 1e-4)
  D_opt = tf.train.AdamOptimizer(4e-4, 0., 0.999, 1e-4)

  # following SAGAN, nD = 1
  G_step = tf.train.get_global_step()
  train_op = tf.group(
    G_opt.minimize(L_G, G_step, G.trainable_weights),
    D_opt.minimize(L_D, var_list=D.trainable_weights))

  # create some tensorboard summaries
  tf.summary.image('xhat', data.postprocess_img(predictions), 5)
  tf.summary.image('x', data.postprocess_img(features), 5)
  tf.summary.scalar('L_G', L_G)
  tf.summary.scalar('L_D', L_D)

  # return an EstimatorSpec
  return tf.estimator.EstimatorSpec(
    mode=mode, loss=L_D, train_op=train_op)

def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)
  estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    params=vars(args),
    config=tf.estimator.RunConfig(
      train_distribute=tf.distribute.MirroredStrategy(
        devices=data.get_gpus()),
      model_dir=args.model_dir))
  estimator.train(input_fn=lambda params: data.get_train_data(
    args.data_file, args.batch_size), max_steps=1000000)

if __name__ == '__main__':
  p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument('data_file', type=str,
    help='.npz file containing preprocessed image data')
  p.add_argument('model_dir', type=str,
    help='directory in which to save checkpoints and summaries')
  p.add_argument('-bs', '--batch_size', type=int, default=64,
    help='number of samples per minibatch update')
  p.add_argument('-ch', '--channels', type=int, default=16,
    help='channel multiplier in G and D')
  p.add_argument('-zd', '--z_dim', type=int, default=128,
    help='dimensionality of latent vector')
  p.add_argument('-dt', '--dtype', choices=('float32', 'float16'),
    default='float16', help='training float precision')
  sys.exit(main(p.parse_args()))
