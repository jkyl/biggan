from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

from layers import *
from keras.models import Model

def resnet_generator(output_size, channels, z_dim):
  z = x = Input((z_dim,))
  l = int(np.log2(output_size)) - 2
  for i in range(l + 1):
    with tf.variable_scope('G_Block_' + str(i + 1)):
      x = g_block(x, int(channels*2**(l-1-i)), first=i==0, last=i==l)
    if i == 3: # 32 x 32
      with tf.variable_scope('G_Attention'):
        x = attention(x)
  return Model(inputs=z, outputs=x)

def resnet_discriminator(input_size, channels):
  inp = x = Input((input_size, input_size, 3))
  l = int(np.log2(input_size)) - 2
  for i in range(l + 1):
    with tf.variable_scope('D_Block_' + str(i + 1)):
      x = d_block(x, int(channels*2**(i if i<l else l-1)), first=i==0, last=i==l)
    if i == l - 4: # 32 x 32
      with tf.variable_scope('D_Attention'):
        x = attention(x)
  return Model(inputs=inp, outputs=x)
