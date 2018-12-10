from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorflow.python.keras.layers import *
from spectral_norm import *

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

def SubPixel(factor, name=None):
  def func(x):
    import tensorflow as tf
    return tf.depth_to_space(x, factor)
  def output_shape(input_shape):
    n, h, w, c = input_shape
    return (n, h * factor, w * factor, c // factor ** 2)
  return Lambda(func, output_shape=output_shape, name=name)

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

def g_block(x, dim, first=False, last=False):
  if first:
    x = DenseSN(4 * 4 * dim, use_bias=False)(x)
    x = Reshape((4, 4, dim))(x)
  else:
    x0 = x
    for j in range(2):
      x = BatchNormalization(axis=-1, scale=False)(x)
      x = Activation('relu')(x)
      if j == 0:
        x = SubPixel(2)(x)
        x0 = SubPixel(2)(x0)
      x = ConvSN2D(dim, 3, padding='same', use_bias=False)(x)
    x0 = ConvSN2D(dim, 1, use_bias=False)(x0)
    x = Add()([x, x0])
  if last:
    x = BatchNormalization(axis=-1, scale=False)(x)
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
  #else:
  #  x = Activation('relu')(x)
  #  x = ConvSN2D(dim, 3, padding='same', kernel_initializer='orthogonal')(x)
  #  x = Activation('relu')(x)
  #  x = ConvSN2D(dim, 3, padding='same', kernel_initializer='orthogonal')(x)
  x = Add()([x, x0])
  if last:
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = DenseSN(1)(x)
  return x
