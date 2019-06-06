from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Dot
from tensorflow.keras import Model
from tensorflow.keras import Input

from .custom_layers import SyncBatchNorm
from .custom_layers import ConvSN2D
from .custom_layers import DenseSN


def _module(function):
  def decorated(*args, **kwargs):
    with tf.compat.v1.variable_scope(None, 
        default_name=function.__name__):
      return function(*args, **kwargs)
  return decorated

def DropChannels(output_dim):
  def call(x):
    return x[..., :output_dim]
  def output_shape(input_shape):
    return input_shape[:-1] + (output_dim,)
  return tf.keras.layers.Lambda(call, output_shape=output_shape)

@_module
def GBlock(x, output_dim, up=False):
  input_dim = int_shape(x)[-1]
  x0 = x
  x = SyncBatchNorm()(x)
  x = Activation('relu')(x)
  x = ConvSN2D(input_dim//4, 1, use_bias=False)(x)
  x = SyncBatchNorm()(x)
  x = Activation('relu')(x)
  if up:
    x = UpSampling2D()(x)
  x = ConvSN2D(input_dim//4, 3, padding='same', use_bias=False)(x)
  x = SyncBatchNorm()(x)
  x = Activation('relu')(x)
  x = ConvSN2D(input_dim//4, 3, padding='same', use_bias=False)(x)
  x = SyncBatchNorm()(x)
  x = Activation('relu')(x)
  x = ConvSN2D(output_dim, 1, use_bias=False)(x)
  if input_dim > output_dim:
    x0 = DropChannels(output_dim)(x0)
  elif input_dim < output_dim:
    raise ValueError
  if up:
    x0 = UpSampling2D()(x0)
  return Add()([x, x0])

@_module
def DBlock(x, output_dim, down=False):
  input_dim = int_shape(x)[-1]
  x0 = x
  x = Activation('relu')(x)
  x = ConvSN2D(input_dim//4, 1)(x)
  x = Activation('relu')(x)
  x = ConvSN2D(input_dim//4, 3, padding='same')(x)
  x = Activation('relu')(x)
  x = ConvSN2D(input_dim//4, 3, padding='same')(x)
  x = Activation('relu')(x)
  if down:
    x = AveragePooling2D()(x)
    x0 = AveragePooling2D()(x0)
  if input_dim < output_dim:
    extra = output_dim - input_dim
    x0_extra = ConvSN2D(extra, 1, use_bias=False)(x0)
    x0 = Concatenate()([x0, x0_extra])
  elif input_dim > output_dim:
    raise ValueError
  x = ConvSN2D(output_dim, 1)(x)
  return Add()([x, x0])

@_module
def Attention(x):
  _b, _h, _w, _c = int_shape(x)
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

  # input z-vector
  z = Input((128,))
  
  # project z and reshape
  x = DenseSN(4 * 4 * 16 * ch, use_bias=False)(z)
  x = Reshape((4, 4, 16 * ch))(x)
  
  # 4x4 -> 8x8
  x = GBlock(x, 16 * ch, up=False)
  x = GBlock(x, 16 * ch, up=True)

  # 8x8 -> 16x16
  x = GBlock(x, 16 * ch, up=False)
  x = GBlock(x, 8 * ch, up=True)

  # 16x16 -> 32x32
  x = GBlock(x, 8 * ch, up=False)
  x = GBlock(x, 8 * ch, up=True)
  
  # 32x32 -> 64x64
  x = GBlock(x, 8 * ch, up=False)
  x = GBlock(x, 4 * ch, up=True)

  # non-local @ 64x64
  x = Attention(x)

  # 64x64 -> 128x128
  x = GBlock(x, 4 * ch, up=False)
  x = GBlock(x, 2 * ch, up=True)

  # 128x128 -> 256x256
  x = GBlock(x, 2 * ch, up=False)
  x = GBlock(x, ch, up=True)

  # output block @ 256x256
  x = SyncBatchNorm()(x)
  x = Activation('relu')(x)
  x = ConvSN2D(3, 3, padding='same')(x)
  x = Activation('tanh')(x)
  
  # return keras model
  return Model(inputs=z, outputs=x, name='Generator')

def Discriminator(ch):
  
  # input image
  x = inp = Input((256, 256, 3))
  
  # 256x256 -> 128x128
  x = ConvSN2D(ch, 3, padding='same')(x)
  x = DBlock(x, 2 * ch, down=True) 
  x = DBlock(x, 2 * ch, down=False)

  # 128x128 -> 64x64
  x = DBlock(x, 4 * ch, down=True)
  x = DBlock(x, 4 * ch, down=False)
 
  # non-local @ 64x64
  x = Attention(x)

  # 64x64 -> 32x32
  x = DBlock(x, 8 * ch, down=True)
  x = DBlock(x, 8 * ch, down=False)

  # 32x32 -> 16x16
  x = DBlock(x, 8 * ch, down=True)
  x = DBlock(x, 8 * ch, down=False)

  # 16x16 -> 8x8
  x = DBlock(x, 16 * ch, down=True)
  x = DBlock(x, 16 * ch, down=False)
  
  # 8x8 -> 4x4
  x = DBlock(x, 16 * ch, down=True)
  x = DBlock(x, 16 * ch, down=False)

  # pool and project to scalar
  x = Activation('relu')(x)
  x = GlobalAveragePooling2D()(x)
  x = DenseSN(1)(x)

  # return keras model
  return Model(inputs=inp, outputs=x, name='Discriminator')
