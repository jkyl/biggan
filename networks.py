from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from layers import *
from tensorflow.keras.models import Model
import numpy as np

def resnet_generator(output_size, channels, z_dim):
  z = Input((z_dim,))
  x = DenseSN(4*4*16*channels)(z)
  x = Reshape((4, 4, 16*channels))(x)
  l = int(np.log2(output_size)) - 2
  for i in range(1, l + 1):
    dim = channels * 2 ** (l - i)
    if i == l:
      x = self_attention(x, dim)
    x = residual_upconv(x, dim)
  x = InstanceNormalization(axis=-1, scale=False)(x)
  x = Activation('relu')(x)
  x = ConvSN2D(3, 3, bias=True)(x)
  x = Activation('tanh')(x)
  return Model(inputs=z, outputs=x)

def resnet_discriminator(input_size, channels):
  inp = x = Input((input_size, input_size, 3))
  l = int(np.log2(input_size)) - 2
  for i in range(l + 1):
    dim = channels * 2 ** (i if i < l else l - 1)
    x = residual_downconv(x, dim, first=i==0, last=i==l)
    if i == 0:
      x = self_attention(x, dim)
  x = Activation('relu')(x)
  x = GlobalAveragePooling2D()(x)
  x = DenseSN(1)(x)
  return Model(inputs=inp, outputs=x)
