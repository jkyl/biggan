from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import argparse
import data
import nets
import sys
import os

class GAN(object):
  def __init__(self, image_size, channels, z_dim):
    self.G = nets.resnet_generator(image_size, channels, z_dim)
    self.D = nets.resnet_discriminator(image_size, channels)

  def hinge_loss(self, x, xhat):
    logits_real = self.D(x)
    logits_fake = self.D(xhat)
    L_G = -tf.reduce_mean(logits_fake)
    L_D = tf.reduce_mean(tf.nn.relu(1 - logits_real))\
        + tf.reduce_mean(tf.nn.relu(1 + logits_fake))
    return L_G, L_D

def model_fn(features, labels, mode, params):
  if mode != tf.estimator.ModeKeys.TRAIN:
    raise NotImplementedError
  model = GAN(
    params['image_size'],
    params['channels'],
    params['z_dim'])
  z = tf.random.normal((
    params['batch_size'],
    params['z_dim']))
  predictions = model.G(z)
  L_G, L_D = model.hinge_loss(features, predictions)
  G_opt = tf.train.AdamOptimizer(1e-4, 0., 0.999)
  D_opt = tf.train.AdamOptimizer(4e-4, 0., 0.999)
  if params['use_tpu']:
    G_opt = tf.contrib.tpu.CrossShardOptimizer(G_opt)
    D_opt = tf.contrib.tpu.CrossShardOptimizer(D_opt)
  step = tf.train.get_global_step()
  with tf.control_dependencies([tf.print('step: ', step)]):
    train_op = tf.cond(
      tf.cast(tf.mod(step, params['n_D'] + 1), tf.bool),
      lambda: G_opt.minimize(L_G,
        var_list=model.G.trainable_weights, global_step=step),
      lambda: D_opt.minimize(L_D,
        var_list=model.D.trainable_weights, global_step=step))
  return tf.contrib.tpu.TPUEstimatorSpec(
    mode=mode, loss=L_D, train_op=train_op)

def input_fn(params):
  return data.get_image_data(
    params['data_dir'],
    params['image_size'],
    params['batch_size'])

def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.contrib.tpu.TPUEstimator(
    model_fn=model_fn,
    use_tpu=args.use_tpu,
    train_batch_size=args.batch_size,
    params=dict(
      data_dir=args.data_dir,
      image_size=args.image_size,
      channels=args.channels,
      use_tpu=args.use_tpu,
      z_dim=args.z_dim,
      n_D=args.n_D),
    config=tf.contrib.tpu.RunConfig(
      cluster=tf.contrib.cluster_resolver.TPUClusterResolver(
        os.environ['TPU_NAME'] if args.use_tpu else ''),
      model_dir=args.model_dir,
      session_config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False),
      tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=10,
        num_shards=8))
  ).train(input_fn=input_fn, max_steps=1000000)

if __name__ == '__main__':
  p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument('data_dir', type=str,
    help='directory containing training PNGs and/or JPGs')
  p.add_argument('model_dir', type=str,
    help='directory in which to save checkpoints and summaries')
  p.add_argument('-tpu', '--use_tpu', action='store_true',
    help='whether or not to use the TPU cluster')
  p.add_argument('-bs', '--batch_size', type=int, default=64,
    help='number of samples per minibatch update')
  p.add_argument('-is', '--image_size', type=int, default=128,
    help='size of generated and real images')
  p.add_argument('-ch', '--channels', type=int, default=64,
    help='channel multiplier in G and D')
  p.add_argument('-zd', '--z_dim', type=int, default=128,
    help='dimensionality of latent vector')
  p.add_argument('-nd', '--n_D', type=int, default=2,
    help='number of D updates per G update')
  sys.exit(main(p.parse_args()))
