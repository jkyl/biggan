from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from sn import ConvSN2D, DenseSN
from bn import SyncBatchNorm
from up import UnPooling2D

def _call(layers, x):
  for f in layers:
    x = f(x)
  return x

class GBlock(tf.keras.Model):
  def __init__(self, dim):
    super(GBlock, self).__init__()
    # main
    self.norm_1 = SyncBatchNorm()
    self.relu_1 = Activation('relu')
    self.unpool = UnPooling2D()
    self.conv_1 = ConvSN2D(dim, 3, padding='same', use_bias=False)
    self.norm_2 = SyncBatchNorm()
    self.relu_2 = Activation('relu')
    self.conv_2 = ConvSN2D(dim, 3, padding='same', use_bias=False)
    # residual
    self.res_project = ConvSN2D(dim, 1, use_bias=False)
    self.res_unpool = UnPooling2D()
  def call(self, x):
    return _call([self.res_project, self.res_unpool], x) \
         + _call([self.norm_1, self.relu_1, self.conv_1, self.unpool,
                  self.norm_2, self.relu_2, self.conv_2], x)

class DBlock(tf.keras.Model):
  def __init__(self, dim, down=True):
    super(DBlock, self).__init__()
    # main
    self.relu_1 = Activation('relu')
    self.conv_1 = ConvSN2D(dim, 3, padding='same')
    self.relu_2 = Activation('relu')
    self.conv_2 = ConvSN2D(dim, 3, padding='same')
    self.pool = AveragePooling2D() if down else lambda x: x
    # residual
    self.res_pool = AveragePooling2D() if down else lambda x: x
    self.res_project = ConvSN2D(dim, 1)
  def call(self, x):
    return _call([self.res_pool, self.res_project], x) \
         + _call([self.relu_1, self.conv_1,
                  self.relu_2, self.conv_2, self.pool], x)

class DBlockInput(tf.keras.Model):
  def __init__(self, dim):
    super(DBlockInput, self).__init__()
    # main
    self.conv_1 = ConvSN2D(dim, 3, padding='same')
    self.relu_1 = Activation('relu')
    self.conv_2 = ConvSN2D(dim, 3, padding='same')
    self.pool = AveragePooling2D()
    # residual
    self.res_pool = AveragePooling2D()
    self.res_project = ConvSN2D(dim, 1)
  def call(self, x):
    return _call([self.res_pool, self.res_project], x) \
         + _call([self.conv_1, self.relu_1,
                  self.conv_2, self.pool], x)

class Attention(tf.keras.Model):
  def __init__(self, dim):
    super(Attention, self).__init__()
    self.project_f = ConvSN2D(dim // 8, 1, use_bias=False)
    self.pool_g = AveragePooling2D()
    self.project_g = ConvSN2D(dim // 8, 1, use_bias=False)
    self.pool_h = AveragePooling2D()
    self.project_h = ConvSN2D(dim // 2, 1, use_bias=False)
    self.project_out = ConvSN2D(dim, 1, use_bias=False)

  def call(self, x):
    _b, _h, _w, _c = K.int_shape(x)
    f = _call([self.project_f], x)
    f = tf.reshape(f, (-1, _h * _w, _c // 8))
    g = _call([self.pool_g, self.project_g], x)
    g = tf.reshape(g, (-1, _h * _w // 4, _c // 8))
    h = _call([self.pool_h, self.project_h], x)
    h = tf.reshape(h, (-1, _h * _w // 4, _c // 2))
    attn = tf.nn.softmax(tf.matmul(f, g, transpose_b=True))
    y = tf.matmul(attn, h)
    y = tf.reshape(y, (-1, _h, _w, _c // 2))
    return _call([self.project_out], y) + x

class Generator(tf.keras.Model):
  def __init__(self, ch):
    super(Generator, self).__init__()
    self.dense = DenseSN(4 * 4 * 16 * ch, use_bias=False)
    self.reshape = Reshape((4, 4, 16 * ch))
    self.block_1 = GBlock(16 * ch)
    self.block_2 = GBlock(8 * ch)
    self.block_3 = GBlock(8 * ch)
    self.block_4 = GBlock(4 * ch)
    self.attn = Attention(4 * ch)
    self.block_5 = GBlock(2 * ch)
    self.block_6 = GBlock(1 * ch)
    self.norm = SyncBatchNorm()
    self.relu = Activation('relu')
    self.conv = ConvSN2D(3, 3, padding='same')
    self.tanh = Activation('tanh')
  def call(self, x):
    return _call([
      self.dense,
      self.reshape,
      self.block_1,
      self.block_2,
      self.block_3,
      self.block_4,
      self.attn,
      self.block_5,
      self.block_6,
      self.norm,
      self.relu,
      self.conv,
      self.tanh,
    ], x)

class Discriminator(tf.keras.Model):
  def __init__(self, ch):
    super(Discriminator, self).__init__()
    self.block_1 = DBlockInput(1 * ch)
    self.block_2 = DBlock(2 * ch)
    self.attn = Attention(2 * ch)
    self.block_3 = DBlock(4 * ch)
    self.block_4 = DBlock(8 * ch)
    self.block_5 = DBlock(8 * ch)
    self.block_6 = DBlock(16 * ch)
    self.block_7 = DBlock(16 * ch, down=False)
    self.relu = Activation('relu')
    self.pool = GlobalAveragePooling2D()
    self.dense = DenseSN(1)
  def call(self, x):
    return _call([
      self.block_1,
      self.block_2,
      self.attn,
      self.block_3,
      self.block_4,
      self.block_5,
      self.block_6,
      self.block_7,
      self.relu,
      self.pool,
      self.dense,
    ], x)
