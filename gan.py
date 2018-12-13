from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.contrib import summary
import argparse
import data
import nets
import sys
import os

def model_fn(features, labels, mode, params):

  # set the learning phase and float precision
  tf.keras.backend.set_learning_phase(True)
  tf.keras.backend.set_floatx(params['dtype'])
  features = tf.cast(features, params['dtype'])

  # build the generator
  G = nets.resnet_generator(
    params['image_size'],
    params['channels'],
    params['z_dim'])

  # build the discriminator
  D = nets.resnet_discriminator(
    params['image_size'],
    params['channels'])

  # sample latent vector `z` from N(0, 1)
  z = tf.random_normal((
    params['train_batch_size'],
    params['z_dim']), dtype=params['dtype'])

  # generate an image from `z`
  predictions = G(z)

  # discriminate real and fake images
  logits_real = D(features)
  logits_fake = D(predictions)

  # "hinge" loss function
  L_G = -tf.reduce_mean(logits_fake)
  L_D = tf.reduce_mean(tf.nn.relu(1 - logits_real))\
      + tf.reduce_mean(tf.nn.relu(1 + logits_fake))

  # two-timescale update rule
  G_opt = tf.train.AdamOptimizer(1e-4, 0., 0.999, 1e-4)
  D_opt = tf.train.AdamOptimizer(4e-4, 0., 0.999, 1e-4)
  if params['use_tpu']:
    G_opt = tf.contrib.tpu.CrossShardOptimizer(G_opt)
    D_opt = tf.contrib.tpu.CrossShardOptimizer(D_opt)

  # every `n_D` steps, update both networks.
  # otherwise, just update the discriminator
  G_step, D_step = tf.train.get_global_step(), tf.Variable(0)
  only_train_D = tf.cast(tf.mod(D_step, params['n_D']), tf.bool)
  def train_G():
    return G_opt.minimize(L_G, G_step, G.trainable_weights)
  def train_D():
    return D_opt.minimize(L_D, D_step, D.trainable_weights)
  def train_both():
    return tf.group(train_G(), train_D())
  train_op = tf.cond(only_train_D, train_D, train_both)

  # save some tensorboard summaries
  def host_call_fn(step, features, predictions, L_G, L_D):
    step = step[0]
    summary.create_file_writer(
      params['model_dir'], flush_millis=1000).set_as_default()
    with summary.always_record_summaries():
      summary.image('xhat', predictions*.5+.5, max_images=5, step=step)
      summary.image('x', features*.5+.5, max_images=5, step=step)
      summary.scalar('L_G', L_G[0], step=step)
      summary.scalar('L_D', L_D[0], step=step)
      return summary.all_summary_ops()
  host_call = (host_call_fn,
    [tf.tile(tf.expand_dims(t, 0), [params['train_batch_size'], 1])
      if not len(t.shape.as_list()) else t
        for t in [G_step, features, predictions, L_G, L_D]])

  # return an EstimatorSpec
  return tf.contrib.tpu.TPUEstimatorSpec(
    mode=mode, loss=L_D, train_op=train_op, host_call=host_call)

def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)
  estimator = tf.contrib.tpu.TPUEstimator(
    model_fn=model_fn,
    params=args.__dict__,
    use_tpu=args.use_tpu,
    train_batch_size=args.train_batch_size,
    config=tf.contrib.tpu.RunConfig(
      model_dir=args.model_dir,
      tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=args.n_per_loop, num_shards=8),
      cluster=tf.contrib.cluster_resolver.TPUClusterResolver(
        os.environ['TPU_NAME']) if args.use_tpu else None))

  estimator.train(input_fn=lambda params:
    data.get_image_data(
      args.data_dir,
      args.image_size,
      args.train_batch_size), max_steps=1000000)

if __name__ == '__main__':
  p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument('data_dir', type=str,
    help='directory containing training PNGs and/or JPGs')
  p.add_argument('model_dir', type=str,
    help='directory in which to save checkpoints and summaries')
  p.add_argument('-tpu', '--use_tpu', action='store_true',
    help='whether to use a TPU cluster')
  p.add_argument('-bs', '--train_batch_size', type=int, default=64,
    help='number of samples per minibatch update')
  p.add_argument('-is', '--image_size', type=int, default=128,
    help='size of generated and real images')
  p.add_argument('-ch', '--channels', type=int, default=64,
    help='channel multiplier in G and D')
  p.add_argument('-zd', '--z_dim', type=int, default=128,
    help='dimensionality of latent vector')
  p.add_argument('-nd', '--n_D', type=int, default=1,
    help='number of D updates per G update')
  p.add_argument('-nl', '--n_per_loop', type=int, default=100,
    help='number of G updates per single execution of train_op')
  p.add_argument('-dt', '--dtype', choices=('float32', 'float16'),
    default='float32', help='training float precision')
  sys.exit(main(p.parse_args()))
