from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorflow.keras.layers import *
from instance_norm import InstanceNormalization
import spectral_norm as spectral

def DenseSN(dim, bias=False):
  return spectral.DenseSN(dim, use_bias=bias, kernel_initializer='orthogonal')

def ConvSN2D(dim, kernel_size, bias=False):
  return spectral.ConvSN2D(dim, kernel_size,
    padding='same', use_bias=bias, kernel_initializer='orthogonal')

def SubPixel(factor, name=None):
  def func(x):
    import tensorflow as tf
    return tf.depth_to_space(x, factor)
  def output_shape(input_shape):
    n, h, w, c = input_shape
    return (n, h * factor, w * factor, c // factor ** 2)
  return Lambda(func, output_shape=output_shape, name=name)

def self_attention(x, dim):
  n, h, w, c = x.shape.as_list()
  theta = ConvSN2D(dim, 1)(x)
  theta = Reshape((-1, dim))(theta)
  phi = ConvSN2D(dim, 1)(x)
  phi = Reshape((-1, dim))(phi)
  f = Dot(2)([theta, phi])
  f = Activation('softmax')(f)
  g = ConvSN2D(dim, 1)(x)
  g = Reshape((-1, dim))(g)
  y = Dot([2, 1])([f, g])
  y = Reshape((h, w, dim))(y)
  y = ConvSN2D(c, 1)(y)
  y = Add()([x, y])
  return y

def residual_upconv(x, dim):
  x0 = x
  for j in range(2):
    x = InstanceNormalization(axis=-1, scale=False)(x)
    x = Activation('relu')(x)
    if j == 0:
      x = SubPixel(2)(x)
      x0 = SubPixel(2)(x0)
    x = ConvSN2D(dim, 3)(x)
  x0 = ConvSN2D(dim, 1)(x0)
  x = Add()([x, x0])
  return x

def residual_downconv(x, dim, first=False, last=False):
  x0 = x
  if first:
    x = ConvSN2D(dim, 3, bias=True)(x)
    x = Activation('relu')(x)
    x = ConvSN2D(dim, 3, bias=True)(x)
    x0 = AveragePooling2D()(x0)
    x0 = ConvSN2D(dim, 1, bias=True)(x0)
    x = AveragePooling2D()(x)
  elif not last:
    x = Activation('relu')(x)
    x = ConvSN2D(dim, 3, bias=True)(x)
    x = Activation('relu')(x)
    x = ConvSN2D(dim, 3, bias=True)(x)
    x0 = ConvSN2D(dim, 1, bias=True)(x0)
    x0 = AveragePooling2D()(x0)
    x = AveragePooling2D()(x)
  else:
    x = Activation('relu')(x)
    x = ConvSN2D(dim, 3, bias=True)(x)
    x = Activation('relu')(x)
    x = ConvSN2D(dim, 3, bias=True)(x)
  x = Add()([x, x0])
  return x
