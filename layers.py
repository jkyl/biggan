from keras.layers import *
from spectral import *

from keras import backend as K
from keras_contrib.layers import InstanceNormalization
from non_local import non_local_block

def UnPooling2D(w, name=None):
  def resize(x):
    x = K.tf.image.resize_nearest_neighbor(x, [w]*2, align_corners=True)
    s = x.shape.as_list()
    x.set_shape([s[0], w, w, s[-1]])
    return x
  return Lambda(resize, name=name)

def conv2d(x, n, k, s=1, norm=False, act=False, res=False, sn=False):
  x = (ConvSN2D if sn else Conv2D)(n, k, strides=max(s, 1), padding='same', use_bias=not norm)(x)
  if norm:
    x = InstanceNormalization(scale=act and not act.endswith('relu'))(x)
  if act=='lrelu':
    x = LeakyReLU(0.2)(x)
  elif act:
    x = Activation(act)(x)
  if type(res) is K.tf.Tensor:
    x = Add()([res, x])
  if s < 1:
    x = Lambda(lambda x: K.tf.image.resize_nearest_neighbor(
      x, K.tf.shape(x)[1:3]*int(1/s), align_corners=True))(x)
  return x

def split_zs(z, n, name=None):
  d = z.shape.as_list()[-1]
  def unstack_with_remainder(x):
    output = []
    for i in range(n):
      start = i * (d // n)
      stop = start + d // n if i + 1 < n else d + 1
      output.append(x[:, start:stop])
    return output
  def output_shape(_):
    si = (None, d // n)
    sf = (None, d // n + d % n)
    rv = [si] * (n - 1) + [sf]
    return rv
  return Lambda(unstack_with_remainder, output_shape=output_shape, name=name)(z)

def reshape_zi(zi, w):
  zi = RepeatVector(w**2)(zi)
  zi = Reshape((w, w, zi.shape.as_list()[-1]))(zi)
  return zi
