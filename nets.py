from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Lambda
from sn import ConvSN2D, DenseSN
from bn import SyncBatchNorm

def _call(layers, x):
  for f in layers:
    x = f(x)
  return x

def UnPooling2D():
  def func(x):
    x = tf.transpose(x, [1, 2, 3, 0])
    x = tf.expand_dims(x, 0)
    x = tf.tile(x, [4, 1, 1, 1, 1])
    x = tf.batch_to_space_nd(x, [2, 2], [[0, 0], [0, 0]])
    x = tf.transpose(x[0], [3, 0, 1, 2])
    return x
  def output_shape(input_shape):
    n, h, w, c = input_shape
    return (n, h * 2, w * 2, c)
  return Lambda(func, output_shape=output_shape)

class G_Block(object):
  def __init__(self, dim):
    self._layers = (
      SyncBatchNorm(),
      Activation('relu'),
      UnPooling2D(),
      ConvSN2D(dim, 3, padding='same', use_bias=False),
      SyncBatchNorm(),
      Activation('relu'),
      ConvSN2D(dim, 3, padding='same', use_bias=False)
    )
    self._residual = (
      UnPooling2D(),
      ConvSN2D(dim, 1, use_bias=False)
    )
  def __call__(self, x):
    return _call(self._layers, x) + _call(self._residual, x)

class D_Block(object):
  def __init__(self, dim, down=True):
    self._layers = (
      Activation('relu'),
      ConvSN2D(dim, 3, padding='same'),
      Activation('relu'),
      ConvSN2D(dim, 3, padding='same'),
      AveragePooling2D() if down else lambda x: x,
    )
    self._residual = (
      ConvSN2D(dim, 1),
      AveragePooling2D() if down else lambda x: x,
    )
  def __call__(self, x):
    return _call(self._layers, x) + _call(self._residual, x)

class D_Block_input(object):
  def __init__(self, dim):
    self._layers = (
      ConvSN2D(dim, 3, padding='same'),
      Activation('relu'),
      ConvSN2D(dim, 3, padding='same'),
      AveragePooling2D(),
    )
    self._residual = (
      AveragePooling2D(),
      ConvSN2D(dim, 1),
    )
  def __call__(self, x):
    return _call(self._layers, x) + _call(self._residual, x)

class Attention(object):
  def __init__(self, dim):
    self._f = (
      ConvSN2D(dim // 8, 1, use_bias=False),
      Reshape((-1, dim // 8)),
    )
    self._g = (
      ConvSN2D(dim // 8, 1, use_bias=False),
      AveragePooling2D(),
      Reshape((-1, dim // 8)),
    )
    self._h = (
      ConvSN2D(dim // 2, 1, use_bias=False),
      AveragePooling2D(),
      Reshape((-1, dim // 2)),
    )
    self._j = (
      ConvSN2D(dim, 1, use_bias=False),
    )
  def __call__(self, x):
    b, h, w, c = x.shape.as_list()
    attn = tf.nn.softmax(tf.matmul(
      _call(self._f, x), _call(self._g, x), transpose_b=True))
    y = tf.matmul(attn, _call(self._h, x))
    y = tf.reshape(y, (b, h, w, c // 2))
    return _call(self._j, y) + x

class Generator(tf.keras.Model):
  def __init__(self, ch):
    super(Generator, self).__init__()
    self._layers = (
      DenseSN(4 * 4 * 16 * ch, use_bias=False),
      Reshape((4, 4, 16 * ch)),
      G_Block(16 * ch),
      G_Block(8 * ch),
      G_Block(8 * ch),
      G_Block(4 * ch),
      Attention(4 * ch),
      G_Block(2 * ch),
      G_Block(1 * ch),
      SyncBatchNorm(),
      Activation('relu'),
      ConvSN2D(3, 3, padding='same'),
      Activation('tanh'),
    )
  def call(self, x):
    return _call(self._layers, x)

class Discriminator(tf.keras.Model):
  def __init__(self, ch):
    super(Discriminator, self).__init__()
    self._layers = (
      D_Block_input(1 * ch),
      D_Block(2 * ch),
      Attention(2 * ch),
      D_Block(4 * ch),
      D_Block(8 * ch),
      D_Block(8 * ch),
      D_Block(16 * ch),
      D_Block(16 * ch, down=False),
      Activation('relu'),
      GlobalAveragePooling2D(),
      DenseSN(1)
    )
  def call(self, x):
    return _call(self._layers, x)
