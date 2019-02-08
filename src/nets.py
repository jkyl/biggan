from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Add
from .sn import ConvSN2D, DenseSN
from .bn import SyncBatchNorm
from .up import UnPooling2D

class GBlock(Model):
  def __init__(self, dim):
    super(GBlock, self).__init__()
    self.dim = dim
  def call(self, x):
    x0 = x
    x = SyncBatchNorm()(x)
    x = Activation('relu')(x)
    x = UnPooling2D()(x)
    x = ConvSN2D(self.dim, 3, padding='same', use_bias=False)(x)
    x = SyncBatchNorm()(x)
    x = Activation('relu')(x)
    x = ConvSN2D(self.dim, 3, padding='same', use_bias=False)(x)
    if self.dim != K.int_shape(x0)[-1]:
      x0 = ConvSN2D(self.dim, 1, use_bias=False)(x0)
    x0 = UnPooling2D()(x0)
    return Add()([x, x0])

class DBlock(Model):
  def __init__(self, dim, down=True, first=False):
    super(DBlock, self).__init__()
    self.dim = dim
    self.down = down
    self.first = first
  def call(self, x):
    x0 = x
    if self.first:
      x = Activation('relu')(x)
    x = ConvSN2D(self.dim, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = ConvSN2D(self.dim, 3, padding='same')(x)
    if self.down:
      x = AveragePooling2D()(x)
      x0 = AveragePooling2D()(x0)
    if self.dim != K.int_shape(x0)[-1]:
      x0 = ConvSN2D(self.dim, 1, use_bias=False)(x0)
    return Add()([x, x0])

class Attention(Model):
  def __init__(self):
    super(Attention, self).__init__()
  def call(self, x):
    _b, _h, _w, _c = K.int_shape(x)
    f = ConvSN2D(_c // 8, 1, use_bias=False)(x)
    f = Reshape((_h * _w, _c // 8))(f)
    g = AveragePooling2D()(x)
    g = ConvSN2D(_c // 8, 1, use_bias=False)(g)
    g = Reshape((_h * _w // 4, _c // 8))(g)
    h = AveragePooling2D()(x)
    h = ConvSN2D(_c // 2, 1, use_bias=False)(h)
    h = Reshape((_h * _w // 4, _c // 2))(h)
    attn = K.softmax(K.batch_dot(f, g, axes=-1))
    y = K.batch_dot(attn, h, axes=(2, 1))
    y = Reshape((_h, _w, _c // 2))(y)
    return ConvSN2D(_c, 1, use_bias=False)(y)

class Generator(Model):
  def __init__(self, ch):
    super(Generator, self).__init__()
    self.ch = ch
  def call(self, x):
    x = DenseSN(4 * 4 * 16 * self.ch, use_bias=False)(x)
    x = Reshape((4, 4, 16 * self.ch))(x)
    x = GBlock(16 * self.ch)(x)
    x = GBlock(8 * self.ch)(x)
    x = GBlock(8 * self.ch)(x)
    x = GBlock(4 * self.ch)(x)
    x = Attention()(x)
    x = GBlock(2 * self.ch)(x)
    x = GBlock(1 * self.ch)(x)
    x = SyncBatchNorm()(x)
    x = Activation('relu')(x)
    x = ConvSN2D(3, 3, padding='same')(x)
    return Activation('tanh')(x)

class Discriminator(Model):
  def __init__(self, ch):
    super(Discriminator, self).__init__()
    self.ch = ch
  def call(self, x):
    x = DBlock(1 * self.ch, first=True)(x)
    x = DBlock(2 * self.ch)(x)
    x = Attention(2 * self.ch)(x)
    x = DBlock(4 * self.ch)(x)
    x = DBlock(8 * self.ch)(x)
    x = DBlock(8 * self.ch)(x)
    x = DBlock(16 * self.ch)(x)
    x = DBlock(16 * self.ch, down=False)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    return DenseSN(1)(x)
