from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from .custom_layers import *

def GBlock(x, dim):
  with tf.variable_scope(None, default_name='GBlock'):
    x0 = x
    x = SyncBatchNorm()(x)
    x = Activation('relu')(x)
    x = UnPooling2D()(x)
    x = ConvSN2D(dim, 3, padding='same', use_bias=False)(x)
    x = SyncBatchNorm()(x)
    x = Activation('relu')(x)
    x = ConvSN2D(dim, 3, padding='same', use_bias=False)(x)
    if dim != K.int_shape(x0)[-1]:
      x0 = ConvSN2D(dim, 1, use_bias=False)(x0)
    x0 = UnPooling2D()(x0)
    return Add()([x, x0])

def DBlock(x, dim, first=False, down=True):
  with tf.variable_scope(None, default_name='DBlock'):
    x0 = x
    if first:
      x = Activation('relu')(x)
    x = ConvSN2D(dim, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = ConvSN2D(dim, 3, padding='same')(x)
    if down:
      x = AveragePooling2D()(x)
      x0 = AveragePooling2D()(x0)
    if dim != K.int_shape(x0)[-1]:
      x0 = ConvSN2D(dim, 1, use_bias=False)(x0)
    return Add()([x, x0])

def Attention(x):
  with tf.variable_scope(None, default_name='Attention'):
    _b, _h, _w, _c = K.int_shape(x)
    f = ConvSN2D(_c // 8, 1, use_bias=False)(x)
    f = Reshape((_h * _w, _c // 8))(f)
    g = AveragePooling2D()(x)
    g = ConvSN2D(_c // 8, 1, use_bias=False)(g)
    g = Reshape((_h * _w // 4, _c // 8))(g)
    h = AveragePooling2D()(x)
    h = ConvSN2D(_c // 2, 1, use_bias=False)(h)
    h = Reshape((_h * _w // 4, _c // 2))(h)
    attn = Dot((2, 2))([f, g])
    attn = Activation('softmax')(attn)
    y = Dot((2, 1))([attn, h])
    y = Reshape((_h, _w, _c // 2))(y)
    return ConvSN2D(_c, 1, use_bias=False)(y)

def Generator(ch):
  z = Input((128,))
  x = DenseSN(4 * 4 * 16 * ch, use_bias=False)(z)
  x = Reshape((4, 4, 16 * ch))(x)
  x = GBlock(x, 16 * ch)
  x = GBlock(x, 8 * ch)
  x = GBlock(x, 8 * ch)
  x = GBlock(x, 4 * ch)
  x = Attention(x)
  x = GBlock(x, 2 * ch)
  x = GBlock(x, 1 * ch)
  x = SyncBatchNorm()(x)
  x = Activation('relu')(x)
  x = ConvSN2D(3, 3, padding='same')(x)
  x = Activation('tanh')(x)
  return Model(inputs=z, outputs=x, name='Generator')

def Discriminator(ch):
  x = inp = Input((256, 256, 3))
  x = DBlock(x, 1 * ch, first=True)
  x = DBlock(x, 2 * ch)
  x = Attention(x)
  x = DBlock(x, 4 * ch)
  x = DBlock(x, 8 * ch)
  x = DBlock(x, 8 * ch)
  x = DBlock(x, 16 * ch)
  x = DBlock(x, 16 * ch, down=False)
  x = Activation('relu')(x)
  x = GlobalAveragePooling2D()(x)
  logits = DenseSN(1)(x)
  return Model(inputs=inp, outputs=logits, name='Discriminator')
