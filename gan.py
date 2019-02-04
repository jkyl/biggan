from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import argparse
import data
import nets
import sys

def model_fn(features, labels, mode, params):
  del labels # unused

  # set the learning phase and float precision
  tf.keras.backend.set_learning_phase(True)
  tf.keras.backend.set_floatx(params['dtype'])
  features = tf.cast(features, params['dtype'])
  if params['dtype'] == 'float16':
    tf.keras.backend.set_epsilon(1e-4)

  # build the generator
  G = nets.resnet_generator(
    params['image_size'],
    params['channels'],
    params['z_dim'])

  # build the discriminator
  D = nets.resnet_discriminator(
    params['image_size'],
    params['channels'])

  # sample z from max(N(0,1), 0)
  z = tf.random_normal((
    params['train_batch_size'],
    params['z_dim']), dtype=params['dtype'])
  z = tf.maximum(z, tf.zeros_like(z))

  # generate image from z
  predictions = G(z)

  # discriminate real and fake images
  logits_real = D(features)
  logits_fake = D(predictions)

  # hinge loss function
  L_G = -tf.reduce_mean(logits_fake)
  L_D = tf.reduce_mean(tf.nn.relu(1. - logits_real))\
      + tf.reduce_mean(tf.nn.relu(1. + logits_fake))

  # two-timescale update rule
  G_opt = tf.train.AdamOptimizer(1e-4, 0., 0.999, tf.keras.backend.epsilon())
  D_opt = tf.train.AdamOptimizer(4e-4, 0., 0.999, tf.keras.backend.epsilon())

  # every n_D steps, update both networks.
  # otherwise, just update the discriminator
  G_step, D_step = tf.train.get_global_step(), tf.Variable(0)
  only_train_D = tf.cast(tf.mod(D_step, params['n_D']), tf.bool)
  def train_G():
    return tf.group(G_opt.minimize(L_G, G_step, G.trainable_weights))
  def train_D():
    return tf.group(D_opt.minimize(L_D, D_step, D.trainable_weights))
  def train_both():
    return tf.group(train_G(), train_D())
  train_op = tf.cond(only_train_D, train_D, train_both)

  # create some tensorboard summaries
  tf.summary.image('xhat', predictions * .5 + .5, 5)
  tf.summary.image('x', features * .5 + .5, 5)
  tf.summary.scalar('L_G', L_G)
  tf.summary.scalar('L_D', L_D)

  # return an TPUEstimatorSpec
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
    args.data_file, args.train_batch_size), max_steps=1000000)

if __name__ == '__main__':
  p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument('data_file', type=str,
    help='.npz file containing preprocessed image data')
  p.add_argument('model_dir', type=str,
    help='directory in which to save checkpoints and summaries')
  p.add_argument('-bs', '--train_batch_size', type=int, default=64,
    help='number of samples per minibatch update')
  p.add_argument('-is', '--image_size', type=int, default=256,
    help='size of generated and real images')
  p.add_argument('-ch', '--channels', type=int, default=16,
    help='channel multiplier in G and D')
  p.add_argument('-zd', '--z_dim', type=int, default=128,
    help='dimensionality of latent vector')
  p.add_argument('-nd', '--n_D', type=int, default=2,
    help='number of D updates per G update')
  p.add_argument('-nl', '--n_per_loop', type=int, default=100,
    help='number of G updates per update loop')
  p.add_argument('-dt', '--dtype', choices=('float32', 'float16'),
    default='float16', help='training float precision')
  sys.exit(main(p.parse_args()))
