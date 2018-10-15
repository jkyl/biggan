from keras.layers import Input, Conv2D, Activation, LeakyReLU, Add, Lambda
from keras_contrib.layers import InstanceNormalization
from keras.models import Model
import keras.backend as K

def conv2d(x, n, k, s=1, norm=True, act='relu', res=False):
  if s >= 1:
    x = Conv2D(n, k, strides=s, padding='same', use_bias=not norm)(x)
  else:
    x = Lambda(lambda x: K.tf.image.resize_nearest_neighbor(x, int(1/s)*K.tf.shape(x)[1:3], align_corners=True))(x)
    x = Conv2D(n, k, strides=1, padding='same', use_bias=not norm)(x)
  if norm:
    x = InstanceNormalization(scale=act and not act.endswith('relu'))(x)
  if act=='lrelu':
    x = LeakyReLU(0.2)(x)
  elif act:
    x = Activation(act)(x)
  if type(res) is K.tf.Tensor:
    x = Add()([res, x])
  return x

def CycleGAN_generator(n_blocks=6):
  inp = x = Input((None, None, 3))
  x = conv2d(x, 32, 7)
  x = conv2d(x, 64, 3, s=2)
  x = conv2d(x, 128, 3, s=2)
  for i in range(n_blocks):
    xi = x
    x = conv2d(x, 128, 3)
    x = conv2d(x, 128, 3, act=None, res=xi)
  x = conv2d(x, 64, 3, s=.5)
  x = conv2d(x, 32, 3, s=.5)
  out = conv2d(x, 3, 7, norm=None, act='tanh')
  return Model(inputs=inp, outputs=out)

def CycleGAN_discriminator(n_layers=4):
  inp = x = Input((None, None, 3))
  for i in range(n_layers):
    n = min(512, 2**(i+6))
    s = 2 if i+1<n_layers else 1
    x = conv2d(x, n, 4, s=s, norm=i, act='lrelu')
  out = conv2d(x, 1, 4, norm=None, act=None)
  return Model(inputs=inp, outputs=out)