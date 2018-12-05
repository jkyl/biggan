from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import argparse
import data
import nets
import sys

class GAN(object):
  def __init__(self, image_size, channels, z_dim):
    self.G = nets.resnet_generator(image_size, channels, z_dim)
    self.D = nets.resnet_discriminator(image_size, channels)
    self.G.summary()
    self.D.summary()

  def hinge_loss(self, x, z):
    logits_real = self.D(x)
    logits_fake = self.D(self.G(z))
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

  def update_fn(i):
    z = tf.random.normal((params['batch_size'], params['z_dim']))
    L_G, L_D = model.hinge_loss(features, z)

    def update_D():
      optim = tf.train.AdamOptimizer(4e-4, 0., 0.999)
      if params['use_tpu']:
        optim = tf.contrib.tpu.CrossShardOptimizer(optim)
      return optim.minimize(L_D)

    def update_G():
      optim = tf.train.AdamOptimizer(1e-4, 0., 0.999)
      if params['use_tpu']:
        optim = tf.contrib.tpu.CrossShardOptimizer(optim)
      return optim.minimize(L_G)

    def update_both():
      return tf.group(update_G(), update_D())

    update_op = tf.cond(
      tf.equal(i, params['n_D'] - 1),
      update_both, update_D)

    with tf.control_dependencies([update_op]):
      return tf.add(i, 1)

  train_op = tf.while_loop(
    lambda i: tf.less(i, params['n_D']),
    update_fn, [tf.constant(0)])

  return tf.contrib.tpu.TPUEstimatorSpec(
    mode=mode, loss=tf.zeros([]), train_op=train_op)

def input_fn(params):
  return data.get_image_data(
    params['data_dir'], params['image_size'], params['batch_size'])

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
      n_D=args.n_D,
    ),
    config=tf.contrib.tpu.RunConfig(
      cluster=tf.contrib.cluster_resolver.TPUClusterResolver(
        os.environ['TPU_NAME'] if args.use_tpu else ''
      ),
      model_dir=args.model_dir,
      session_config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True
      ),
      tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=10,
        num_shards=8
      )
    )
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
