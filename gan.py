from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import networks as nets
import data

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
tf.flags.DEFINE_string('data_dir', './', 'path to directory containing the dataset')
tf.flags.DEFINE_string('model_dir', None, 'estimator model_dir')
tf.flags.DEFINE_integer('batch_size', 64, 'batch size for training')
tf.flags.DEFINE_integer('image_size', 256, 'sidelength of images')
tf.flags.DEFINE_integer('channels', 32, 'channel multiplier for G and D')
tf.flags.DEFINE_integer('z_dim', 128, 'dimensionality of latent vector')
tf.flags.DEFINE_integer('train_steps', 1000000, 'total number of training steps')
tf.flags.DEFINE_integer('iterations_per_loop', 10, 'num. steps per summary')
tf.flags.DEFINE_integer('num_shards', 8, 'number of shards')
FLAGS = tf.flags.FLAGS

class GAN(object):
  def __init__(self, image_size, channels, z_dim, **kwargs):
    print('entering init...')
    self.G, self.D = (
      nets.resnet_generator(image_size, channels, z_dim),
      nets.resnet_discriminator(image_size, channels))
    self.graph = tf.get_default_graph()
    self.image_size = image_size
    self.channels = channels
    self.z_dim = z_dim
    self.G.summary()
    self.D.summary()

  def hinge_loss(self, x, xhat):
    logits_fake, logits_real = self.D(xhat), self.D(x)
    L_G = -tf.reduce_mean(logits_fake)
    L_D = tf.reduce_mean(tf.nn.relu(1 - logits_real))\
        + tf.reduce_mean(tf.nn.relu(1 + logits_fake))
    return L_G, L_D

def model_fn(features, labels, mode, params):
  model = GAN(params['image_size'], params['channels'], params['z_dim'])
  predictions = model.G(tf.random.normal(shape=(params['batch_size'], params['z_dim'])))
  if mode == tf.estimator.ModeKeys.TRAIN:
    step = tf.train.get_global_step()
    L_G, L_D = model.hinge_loss(features, predictions)
    G_opt = tf.contrib.tpu.CrossShardOptimizer(tf.train.AdamOptimizer(
      1e-4, 0., 0.999)).minimize(L_G,
        var_list=model.G.trainable_weights, global_step=step)
    D_opt = tf.contrib.tpu.CrossShardOptimizer(tf.train.AdamOptimizer(
      4e-4, 0., 0.999)).minimize(L_D, var_list=model.D.trainable_weights)
    cond = lambda i: tf.less(i, 2)
    body = lambda i: tf.tuple([tf.add(i, 1), D_opt])[0]
    D_opt = tf.while_loop(cond, body, [tf.constant(0)])
    train_op = tf.group(G_opt, D_opt)
    #def host_call(step, x, xhat, L_G, L_D):
    #  with tf.contrib.summary.create_file_writer(
    #    FLAGS.model_dir, max_queue=FLAGS.iterations_per_loop).as_default():
    #    with tf.contrib.summary.always_record_summaries():
    #      tf.contrib.summary.scalar('L_D', L_D, step=step)
    #      tf.contrib.summary.scalar('L_G', L_G, step=step)
    #      tf.contrib.summary.image('x', x, max_images=5, step=step)
    #      tf.contrib.summary.image('xhat', xhat, max_images=5, step=step)
    #      return tf.contrib.summary.all_summary_ops()
    #x, xhat = [data.postprocess_img(i) for i in (features, predictions)]
    #host_call = (host_call, [step, x, xhat, L_G, L_D])
    return tf.contrib.tpu.TPUEstimatorSpec(
      mode=mode, loss=L_D, train_op=train_op)#, host_call=host_call)
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
      tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=FLAGS.iterations_per_loop,
        num_shards=FLAGS.num_shards),
  )
  estimator = tf.contrib.tpu.TPUEstimator(
      model_fn=model_fn,
      use_tpu=True,
      train_batch_size=FLAGS.batch_size,
      params=dict(
        data_dir=FLAGS.data_dir,
        image_size=FLAGS.image_size,
        channels=FLAGS.channels,
        z_dim=FLAGS.z_dim),
      config=run_config)
  estimator.train(input_fn=input_fn, max_steps=FLAGS.train_steps)

if __name__ == '__main__':
  tf.app.run()
