import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from .custom_layers import *

def GBlock(x, output_dim, up=False):
  input_dim = K.int_shape(x)[-1]
  x0 = x
  x = SyncBatchNorm()(x)
  x = Activation('relu')(x)
  x = ConvSN2D(input_dim//4, 1, use_bias=False)(x)
  x = SyncBatchNorm()(x)
  x = Activation('relu')(x)
  if up:
    x = UnPooling2D()(x)
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
    x0 = UnPooling2D()(x0)
  return Add()([x, x0])

def DBlock(x, output_dim, down=False):
  input_dim = K.int_shape(x)[-1]
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

def Attention(x):
  with tf.compat.v1.variable_scope(None, default_name='Attention'):
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

  # Input `z`-vector
  z = Input((128,))
  
  # Project `z` and reshape
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

  # Non-local @ 64x64
  x = Attention(x)

  # 64x64 -> 128x128
  x = GBlock(x, 4 * ch, up=False)
  x = GBlock(x, 2 * ch, up=True)

  # 128x128 -> 256x256
  x = GBlock(x, 2 * ch, up=False)
  x = GBlock(x, ch, up=True)

  # Output block @ 256x256
  x = SyncBatchNorm()(x)
  x = Activation('relu')(x)
  x = ConvSN2D(3, 3, padding='same')(x)
  x = Activation('tanh')(x)
  
  # Return Keras model
  return Model(inputs=z, outputs=x, name='Generator')

def Discriminator(ch):
  
  # Input image
  x = inp = Input((256, 256, 3))
  
  # 256x256 -> 128x128
  x = ConvSN2D(ch, 3, padding='same')(x)
  x = DBlock(x, 2 * ch, down=True) 
  x = DBlock(x, 2 * ch, down=False)

  # 128x128 -> 64x64
  x = DBlock(x, 4 * ch, down=True)
  x = DBlock(x, 4 * ch, down=False)
 
  # Non-local @ 64x64
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

  # Pool and project to scalar
  x = Activation('relu')(x)
  x = GlobalAveragePooling2D()(x)
  x = DenseSN(1)(x)

  # Return Keras model
  return Model(inputs=inp, outputs=x, name='Discriminator')
