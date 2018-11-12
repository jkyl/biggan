from layers import *
from keras.models import Model

def CycleGAN_generator(n_blocks=6):
  inp = x = Input((None, None, 3))
  x = conv2d(x, 32, 7, act='relu', norm=True, sn=True)
  x = conv2d(x, 64, 3, s=2, act='relu', norm=True, sn=True)
  x = conv2d(x, 128, 3, s=2, act='relu', norm=True, sn=True)
  for i in range(n_blocks):
    xi = x
    x = conv2d(x, 128, 3, act='relu', norm=True, sn=True)
    x = conv2d(x, 128, 3, act=False, norm=True, sn=True, res=xi)
  x = conv2d(x, 64, 3, s=.5, act='relu', norm=True, sn=True)
  x = conv2d(x, 32, 3, s=.5, act='relu', norm=True, sn=True)
  out = conv2d(x, 3, 7, act='tanh', norm=False, sn=True)
  return Model(inputs=inp, outputs=out)

def CycleGAN_discriminator(n_layers=4):
  inp = x = Input((None, None, 3))
  for i in range(n_layers):
    n = min(512, 2**(i+6))
    s = 2 if i+1<n_layers else 1
    x = conv2d(x, n, 4, s=s, norm=False, act='lrelu', sn=True)
  out = conv2d(x, 1, 4, norm=False, act=False, sn=True)
  return Model(inputs=inp, outputs=out)

def resnet_generator(output_size, channels, z_dim):
  z = Input((z_dim,))
  l = int(np.log2(output_size)) - 2
  zs = split_zs(z, l+1, name='split_zs')
  x = Dense(4*4*16*channels, use_bias=False, kernel_initializer='orthogonal')(zs[0])
  x = Reshape((4, 4, 16*channels))(x)
  for i in range(1, l+1):
    x0 = x
    w = 2**(i+1)
    n = channels*2**(l-i)
    for j in range(2):
      x = InstanceNormalization(axis=-1, scale=False)(x)
      x = Activation('relu')(x)
      if j == 0:
        zi = reshape_zi(zs[i], w)
        x = Concatenate()([x, zi])
        x = UnPooling2D(2*w, name='unpooling_'+str(2*i+1))(x)
        x0 = UnPooling2D(2*w, name='unpooling_'+str(2*i+2))(x0)
      x = Conv2D(n, 3, padding='same', use_bias=False, kernel_initializer='orthogonal')(x)
    x0 = Conv2D(n, 3, padding='same', use_bias=False, kernel_initializer='orthogonal')(x0)
    x = Add()([x, x0])
  x = InstanceNormalization(axis=-1, scale=False)(x)
  x = Activation('relu')(x)
  x = Conv2D(3, 3, padding='same', kernel_initializer='orthogonal')(x)
  x = Activation('tanh')(x)
  return Model(inputs=z, outputs=x)

def resnet_discriminator(input_size, channels):
  inp = x = Input((input_size, input_size, 3))
  l = int(np.log2(input_size)) - 2
  for i in range(l):
    x0 = x
    n = channels*2**i
    for j in range(2):
      x = Activation('relu')(x)
      if j == 1:
        x = AveragePooling2D()(x)
        x0 = AveragePooling2D()(x0)
      x = ConvSN2D(n, 3, padding='same', kernel_initializer='orthogonal')(x)
    x0 = ConvSN2D(n, 3, padding='same', kernel_initializer='orthogonal')(x0)
    x = Add()([x, x0])
  x = Activation('relu')(x)
  x = GlobalAveragePooling2D()(x)
  x = DenseSN(1, kernel_initializer='orthogonal')(x)
  return Model(inputs=inp, outputs=x)