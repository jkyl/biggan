from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import networks as nets
import data

import tensorflow as tf
import argparse
import tqdm
import os

class GAN(object):
  def __init__(self, image_size, channels, z_dim, **kwargs):
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

  def prepare_summary(self, output_dir, **kwargs):
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    writer = tf.summary.FileWriter(output_dir, graph=self.graph)
    for image in ['x', 'xhat']:
      if image in kwargs:
        tf.summary.image(image, data.postprocess_img(kwargs[image]), 5)
    for scalar in ['L_G', 'L_D', 'R']:
      if scalar in kwargs:
        tf.summary.scalar(scalar, kwargs[scalar])
    summarize = tf.summary.merge_all()
    return writer, summarize

  def save(self, output_dir, i):
    self.G.save(os.path.join(output_dir, 'G_'+str(i).zfill(8)+'.h5'))

  def train(self, input_dir, output_dir,
      batch_size=16, n_D_updates=1, **kwargs):
    with tf.name_scope('data'):
      coord = tf.train.Coordinator()
      x = data.get_image_data(input_dir, self.image_size, batch_size)
      z = tf.random_normal(shape=(batch_size, self.z_dim))
    with tf.name_scope('losses'):
      xhat = self.G(z)
      L_G, L_D = self.hinge_loss(x, xhat)
    with tf.name_scope('optimizers'):
      G_opt = tf.train.AdamOptimizer(1e-4, 0., 0.999)\
        .minimize(L_G, var_list=self.G.trainable_weights)
      D_opt = tf.train.AdamOptimizer(4e-4, 0., 0.999)\
        .minimize(L_D, var_list=self.D.trainable_weights)
    writer, summarize = self.prepare_summary(output_dir,
      L_G=L_G, L_D=L_D, x=x, xhat=xhat)
    self.save(output_dir, 0)
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    with tf.Session() as sess:
      tf.train.start_queue_runners(sess=sess, coord=coord)
      sess.run(tf.global_variables_initializer())
      self.graph.finalize()
      progbar = tqdm.trange(int(1e6), disable=False)
      for i in progbar:
        for _ in range(n_D_updates - 1):
          sess.run(D_opt, options=run_options)
        if i % 10:
          ld, lg = sess.run([D_opt, G_opt, L_D, L_G], options=run_options)[-2:]
        else:
          ld, lg, s = sess.run([D_opt, G_opt, L_D, L_G, summarize], options=run_options)[-3:]
          writer.add_summary(s, i)
        progbar.set_description('(L_G={:.2f}, L_D={:.2f})'.format(lg, ld))
        if not i + 1 % 10000:
          self.save(output_dir, i + 1)

if __name__ == '__main__':
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  tf.logging.set_verbosity(tf.logging.ERROR)
  p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument('input_dir', type=str,
    help='path containing training images')
  p.add_argument('output_dir', type=str,
    help='path in which to save weights and summaries')
  p.add_argument('-is', '--image_size', type=int, default=128,
    help='real and generated image size')
  p.add_argument('-ch', '--channels', type=int, default=16,
    help='channel multiplier in D and G')
  p.add_argument('-zd', '--z_dim', type=int, default=128,
    help='dimensionality of z-vector')
  p.add_argument('-bs', '--batch_size', type=int, default=16,
    help='number of samples per gradient update')
  p.add_argument('-nd', '--n_D_updates', type=int, default=1,
    help='number of D updates per G update')
  kwargs = p.parse_args().__dict__
  model = GAN(**kwargs).train(**kwargs)
