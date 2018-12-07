from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import argparse
import data
import nets
import sys
import os

def model_fn(features, labels, mode, params):
  G = nets.resnet_generator(
    params['image_size'],
    params['channels'],
    params['z_dim'])
  D = nets.resnet_discriminator(
    params['image_size'],
    params['channels'])
  z = tf.random_normal((
    params['batch_size'],
    params['z_dim']))
  predictions = G(z)
  logits_real = D(features)
  logits_fake = D(predictions)
  L_G = -tf.reduce_mean(logits_fake)
  L_D = tf.reduce_mean(tf.nn.relu(1 - logits_real))\
      + tf.reduce_mean(tf.nn.relu(1 + logits_fake))
  G_opt = tf.train.AdamOptimizer(1e-4, 0., 0.999)
  D_opt = tf.train.AdamOptimizer(4e-4, 0., 0.999)
  if params['use_tpu']:
    G_opt = tf.contrib.tpu.CrossShardOptimizer(G_opt)
    D_opt = tf.contrib.tpu.CrossShardOptimizer(D_opt)
  step, global_step = \
    tf.Variable(0), tf.train.get_global_step()
  train_G = tf.logical_not(tf.cast(tf.mod(
    step, params['n_D'] + 1), tf.bool))
  with tf.control_dependencies([tf.assign_add(
      global_step, tf.cast(train_G, tf.int64))]):
    train_op = tf.cond(train_G,
      lambda: G_opt.minimize(L_G,
        var_list=G.trainable_weights, global_step=step),
      lambda: D_opt.minimize(L_D,
        var_list=D.trainable_weights, global_step=step))
  spec = tf.estimator.EstimatorSpec if not params['use_tpu']\
    else tf.contrib.tpu.TPUEstimatorSpec
  return spec(mode=mode, loss=L_D, train_op=train_op)

def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)
  if args.use_tpu:
    estimator = tf.contrib.tpu.TPUEstimator(
      model_fn=model_fn,
      use_tpu=args.use_tpu,
      train_batch_size=args.batch_size,
      params=args.__dict__,
      config=tf.contrib.tpu.RunConfig(
        cluster=tf.contrib.cluster_resolver.TPUClusterResolver(
          os.environ['TPU_NAME'] if args.use_tpu else ''),
        model_dir=args.model_dir,
        session_config=tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=False),
        tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=10, num_shards=8)))
  else:
    estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=args.model_dir,
      params=args.__dict__)

  estimator.train(input_fn=lambda *_:
    data.get_image_data(
      args.data_dir,
      args.image_size,
      args.batch_size), max_steps=1000000)

if __name__ == '__main__':
  p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument('data_dir', type=str,
    help='directory containing training PNGs and/or JPGs')
  p.add_argument('model_dir', type=str,
    help='directory in which to save checkpoints and summaries')
  p.add_argument('-tpu', '--use_tpu', action='store_true',
    help='whether to use a TPU cluster')
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
