import os
import tqdm
import argparse
import tensorflow as tf
import keras.backend as K
import networks as nets
import data

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

  def hinge_loss(self, x, xhat):
    logits_fake, logits_real = self.D(xhat), self.D(x)
    L_G = -tf.reduce_mean(logits_fake)
    L_D = tf.reduce_mean(tf.nn.relu(1 - logits_real))\
        + tf.reduce_mean(tf.nn.relu(1 + logits_fake))
    return L_G, L_D

  def orthogonal_regularization(self):
    def R(W):
      d = W.shape.as_list()[-1]
      W = tf.reshape(W, (-1, d))
      WTW = tf.matmul(tf.transpose(W), W)
      offdiagonal = 1. - tf.eye(d)
      return tf.nn.l2_loss(WTW * offdiagonal)
    return tf.reduce_sum([R(W)
      for W in self.G.trainable_weights if 'kernel' in W.name])

  def save(self, output_dir, i):
    self.G.save(os.path.join(output_dir, 'G_'+str(i).zfill(8)+'.h5'))

  def train(self, input_dir, output_dir, batch_size=16, n_D_updates=1, **kwargs):
    with tf.name_scope('data'):
      coord = tf.train.Coordinator()
      x = data.get_image_data([input_dir], self.image_size, batch_size, mode='resize')[0]
      z = tf.random_normal(shape=(batch_size, self.z_dim))
    with tf.name_scope('losses'):
      xhat = self.G(z)
      L_G, L_D = self.hinge_loss(x, xhat)
      R = self.orthogonal_regularization()
    with tf.name_scope('optimizers'):
      G_opt = tf.train.AdamOptimizer(2e-4, 0.5, 0.999)\
        .minimize(L_G, var_list=self.G.trainable_weights)
      D_opt = tf.train.AdamOptimizer(2e-4, 0.5, 0.999)\
        .minimize(L_D, var_list=self.D.trainable_weights)
    for image in ['x', 'xhat']:
      tf.summary.image(image, data.postprocess_img(eval(image)), 5)
    for scalar in ['L_G', 'L_D', 'R']:
      tf.summary.scalar(scalar, eval(scalar))
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    writer = tf.summary.FileWriter(output_dir, graph=self.graph)
    summarize = tf.summary.merge_all()
    with K.get_session() as sess:
      tf.train.start_queue_runners(sess=sess, coord=coord)
      sess.run(tf.global_variables_initializer())
      for i in tqdm.trange(int(1e6), disable=False):
        if not i % 10000:
          self.save(output_dir, i)
        for _ in range(n_D_updates - 1):
          sess.run(D_opt)
        if not i % 10:
          s = sess.run([D_opt, G_opt, summarize])[-1]
          writer.add_summary(s, i)
        else:
          sess.run([D_opt, G_opt])

if __name__ == '__main__':
  p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument('input_dir', type=str, help='path containing images from domain `X`')
  p.add_argument('output_dir', type=str, help='path in which to save checkpoints and summaries')
  p.add_argument('-is', '--image_size', type=int, default=128, help='number of pixels per image H & W')
  p.add_argument('-ch', '--channels', type=int, default=16, help='channel multiplier in D and G')
  p.add_argument('-zd', '--z_dim', type=int, default=128, help='dimensionality of latent vector')
  p.add_argument('-bs', '--batch_size', type=int, default=16, help='number of examples per gradient update')
  p.add_argument('-nd', '--n_D_updates', type=int, default=1, help='number of D updates per G update')
  kwargs = p.parse_args().__dict__
  model = GAN(**kwargs).train(**kwargs)
