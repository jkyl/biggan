from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *
from spectral_norm import *

config = {
  512: [16, 16, 8, 8, 4, 2, 1, 1],
  256: [16, 16, 8, 8, 4, 2, 1],
  128: [16, 16, 8, 4, 2, 1],
   64: [16, 8, 4, 2, 1]
}

def resnet_generator(output_size, channels, z_dim):
  z = x = Input((z_dim,))
  l = int(np.log2(output_size))
  for i, n in enumerate(config[output_size]):
    res = 2 ** (2 + i)
    with tf.variable_scope('G_Block_'+str(i+1)):
      x = g_block(x, n*channels, first=i==0, last=i==l-2)
    if res == 32:
      with tf.variable_scope('G_Attention'):
        x = attention(x)
  return Model(inputs=z, outputs=x)

def resnet_discriminator(input_size, channels):
  inp = x = Input((input_size, input_size, 3))
  l = int(np.log2(input_size))
  for i, n in enumerate(reversed(config[input_size])):
    res = 2 ** (l - (i + 1))
    with tf.variable_scope('D_Block_'+str(i+1)):
      x = d_block(x, n*channels, first=i==0, last=i==l-2)
    if res == 32:
      with tf.variable_scope('D_Attention'):
        x = attention(x)
  return Model(inputs=inp, outputs=x)

def g_block(x, dim, first=False, last=False):
  if first:
    x = DenseSN(4 * 4 * dim, use_bias=False)(x)
    x = Reshape((4, 4, dim))(x)
  else:
    eps = tf.keras.backend.epsilon()
    x0 = x
    x = BatchNormalization(axis=-1, scale=False, epsilon=eps)(x)
    x = Activation('relu')(x)
    x = UnPooling2D(2)(x)
    x = ConvSN2D(dim, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization(axis=-1, scale=False, epsilon=eps)(x)
    x = Activation('relu')(x)
    x = ConvSN2D(dim, 3, padding='same', use_bias=False)(x)
    x0 = UnPooling2D(2)(x0)
    x0 = ConvSN2D(dim, 1, use_bias=False)(x0)
    x = Add()([x, x0])
  if last:
    x = BatchNormalization(axis=-1, scale=False, epsilon=eps)(x)
    x = Activation('relu')(x)
    x = ConvSN2D(3, 3, padding='same')(x)
    x = Activation('tanh')(x)
  return x

def d_block(x, dim, first=False, last=False):
  x0 = x
  if first:
    x = ConvSN2D(dim, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = ConvSN2D(dim, 3, padding='same')(x)
    x0 = AveragePooling2D()(x0)
    x0 = ConvSN2D(dim, 1)(x0)
    x = AveragePooling2D()(x)
  elif not last:
    x = Activation('relu')(x)
    x = ConvSN2D(dim, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = ConvSN2D(dim, 3, padding='same')(x)
    x0 = ConvSN2D(dim, 1)(x0)
    x0 = AveragePooling2D()(x0)
    x = AveragePooling2D()(x)
  x = Add()([x, x0])
  if last:
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = DenseSN(1)(x)
  return x

def attention(x):
  n, h, w, c = x.shape.as_list()
  theta = ConvSN2D(c // 8, 1, use_bias=False)(x)
  theta = Reshape((-1, c // 8))(theta)
  phi = ConvSN2D(c // 8, 1, use_bias=False)(x)
  phi = AveragePooling2D()(phi)
  phi = Reshape((-1, c // 8))(phi)
  f = Dot(2)([theta, phi])
  f = Activation('softmax')(f)
  g = ConvSN2D(c // 2, 1, use_bias=False)(x)
  g = AveragePooling2D()(g)
  g = Reshape((-1, c // 2))(g)
  y = Dot([2, 1])([f, g])
  y = Reshape((h, w, c // 2))(y)
  y = ConvSN2D(c, 1, use_bias=False)(y)
  y = Gain()(y)
  y = Add()([x, y])
  return y

def UnPooling2D(factor, name=None):
  def func(x):
    import tensorflow as tf
    x = tf.transpose(x, [1, 2, 3, 0])
    x = tf.expand_dims(x, 0)
    x = tf.tile(x, [factor**2, 1, 1, 1, 1])
    x = tf.batch_to_space_nd(x, [factor, factor], [[0, 0], [0, 0]])
    x = tf.transpose(x[0], [3, 0, 1, 2])
    return x
  def output_shape(input_shape):
    n, h, w, c = input_shape
    return (n, h * factor, w * factor, c)
  return Lambda(func, output_shape=output_shape, name=name)

class Gain(Layer):
  def __init__(self, **kwargs):
    super(Gain, self).__init__(**kwargs)
  def build(self, input_shape):
    self.gamma = self.add_weight(
      name='gamma', shape=[], initializer='zeros', trainable=True)
    super(Gain, self).build(input_shape)
  def call(self, x):
    return self.gamma * x
  def compute_output_shape(self, input_shape):
    return input_shape
