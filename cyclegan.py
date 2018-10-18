import tqdm, argparse
import tensorflow as tf
import keras.backend as K
from networks import *
from data import *

class CycleGAN(object):
  def __init__(self):
    self.F, self.G, self.D_X, self.D_Y = (
      CycleGAN_generator(), CycleGAN_generator(),
      CycleGAN_discriminator(), CycleGAN_discriminator())
    self.graph = tf.get_default_graph()

  def cycle_loss(self, x, y, G_of_x, F_of_y):
    return (tf.reduce_mean(tf.abs(x - self.F(G_of_x)))
          + tf.reduce_mean(tf.abs(y - self.G(F_of_y))))

  def identity_loss(self, x, y, G_of_x, F_of_y):
    return (tf.reduce_mean(tf.abs(x - G_of_x))
          + tf.reduce_mean(tf.abs(y - F_of_y)))

  def lsgan_loss(self, x, y, G_of_x, F_of_y):
    D_X_of_x, D_Y_of_y = (
      tf.sigmoid(self.D_X(x)),
      tf.sigmoid(self.D_Y(y)))
    D_X_of_F_of_y, D_Y_of_G_of_x = (
      tf.sigmoid(self.D_X(F_of_y)),
      tf.sigmoid(self.D_Y(G_of_x)))
    L_G = tf.reduce_mean(
        (D_X_of_F_of_y - 1) ** 2
      + (D_Y_of_G_of_x - 1) ** 2)
    L_D = tf.reduce_mean(
        (D_X_of_x - 1) ** 2
      + (D_Y_of_y - 1) ** 2
      + D_X_of_F_of_y ** 2
      + D_Y_of_G_of_x ** 2)
    return L_G, L_D

  def save(self, output_dir, i):
    suffix = os.path.join(output_dir, '{}_'+str(i).zfill(8)+'.h5')
    self.G.save(suffix.format('G'))
    self.F.save(suffix.format('F'))

  def train(self, X_dir, Y_dir, output_dir, crop_size=256, batch_size=1, lambda_c=10, lambda_i=1):
    with tf.name_scope('data'):
      coord = tf.train.Coordinator()
      x, y = get_image_data([X_dir, Y_dir], crop_size, batch_size)
    with tf.name_scope('losses'):
      G_of_x, F_of_y = self.G(x), self.F(y)
      L_cyc = self.cycle_loss(x, y, G_of_x, F_of_y)
      L_idy = self.identity_loss(x, y, G_of_x, F_of_y)
      L_adv_G, L_D = self.lsgan_loss(x, y, G_of_x, F_of_y)
      L_G = lambda_c * L_cyc + lambda_i * L_idy + L_adv_G
    with tf.name_scope('optimizers'):
      G_opt = tf.train.AdamOptimizer(1e-4).minimize(
        L_G, var_list=self.F.trainable_weights+self.G.trainable_weights)
      D_opt = tf.train.AdamOptimizer(1e-4).minimize(
        L_D, var_list=self.D_X.trainable_weights+self.D_Y.trainable_weights)
    for image in ['x', 'y', 'G_of_x', 'F_of_y']:
      tf.summary.image(image, postprocess_img(eval(image)), 5)
    for scalar in ['L_adv_G', 'L_cyc', 'L_D', 'L_idy']:
      tf.summary.scalar(scalar, eval(scalar))
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    writer = tf.summary.FileWriter(output_dir, graph=self.graph)
    summary_op = tf.summary.merge_all()
    with K.get_session() as sess:
      tf.train.start_queue_runners(sess=sess, coord=coord)
      sess.run(tf.global_variables_initializer())
      for i in tqdm.trange(int(1e6), disable=False):
        if i % 10:
          sess.run([G_opt, D_opt])
        else:
          s = sess.run([G_opt, D_opt, summary_op])[-1]
          writer.add_summary(s, i)
        if not i % 10000:
          self.save(output_dir, i)

if __name__ == '__main__':
  p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument('X_dir', type=str, help='path containing images from domain `X`')
  p.add_argument('Y_dir', type=str, help='path containing images from domain `Y`')
  p.add_argument('output_dir', type=str, help='path in which to save checkpoints and summaries')
  p.add_argument('-lc', '--lambda_c', type=float, default=10., help='weight of cycle loss')
  p.add_argument('-li', '--lambda_i', type=float, default=1., help='weight of identity loss')
  kwargs = p.parse_args().__dict__
  model = CycleGAN()
  model.train(**kwargs)