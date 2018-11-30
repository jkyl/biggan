from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import data
import gan

tf.flags.DEFINE_string(
    'tpu', default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.')
tf.flags.DEFINE_string(
    'tpu_zone', default=None,
    help='[Optional] GCE zone where the Cloud TPU is located in. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')
tf.flags.DEFINE_string(
    'gcp_project', default=None,
    help='[Optional] Project name for the Cloud TPU-enabled project. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')
tf.flags.DEFINE_string('data_dir', './', 'Path to directory containing the dataset')
tf.flags.DEFINE_string('model_dir', None, 'Estimator model_dir')
tf.flags.DEFINE_integer('batch_size', 1024,
                        'Mini-batch size for the training. Note that this '
                        'is the global batch size and not the per-shard batch.')
tf.flags.DEFINE_integer('image_size', 256)
tf.flags.DEFINE_integer('z_dim', 128)
tf.flags.DEFINE_integer('train_steps', 1000000, 'Total number of training steps.')
tf.flags.DEFINE_integer('num_shards', 8, 'Number of shards (TPU chips).')
FLAGS = tf.flags.FLAGS

def model_fn(features, labels, mode, params):
  model = gan.GAN(params['image_size'], params['channels'], params['z_dim'])
  predictions = model.G(tf.random.normal(shape=(params['batch_size'], params['z_dim'])))
  if mode == tf.estimator.ModeKeys.TRAIN:
    L_G, L_D = model.hinge_loss(features, predictions)
    optimizer = tf.group(
      tf.contrib.tpu.CrossShardOptimizer(tf.train.AdamOptimizer(
        1e-4, 0., 0.999)).minimize(L_G, model.G.trainable_weights),
      tf.contrib.tpu.CrossShardOptimizer(tf.train.AdamOptimizer(
        4e-4, 0., 0.999)).minimize(L_D, model.D.trainable_weights))
    return tf.contrib.tpu.TPUEstimatorSpec(
      mode=mode, loss=L_D, train_op=optimizer)
  raise NotImplementedError

def input_fn(params):
  return data.get_image_data(
    params['data_dir'], params['image_size'], params['batch_size'])

def main(argv):
  del argv  # Unused.
  tf.logging.set_verbosity(tf.logging.INFO)
  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu,
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project
  )
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=tf.contrib.tpu.TPUConfig(num_shards=FLAGS.num_shards),
  )
  estimator = tf.contrib.tpu.TPUEstimator(
      model_fn=model_fn,
      use_tpu=True,
      train_batch_size=FLAGS.batch_size,
      params={'data_dir': FLAGS.data_dir},
      config=run_config)
  estimator.train(input_fn=input_fn, max_steps=FLAGS.train_steps)

if __name__ == '__main__':
  tf.app.run()
