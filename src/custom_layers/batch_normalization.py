from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

class HyperBatchNorm(tf.keras.layers.Layer):
  '''Cross-replica batch norm layer with scale and
  bias params modulated by auxiliary vector input
  '''
  def __init__(
      self,
      center=True,
      scale=True,
      trainable=True,
      epsilon=1e-3,
      name=None,
      **kwargs
    ):
    super(HyperBatchNorm, self).__init__(
      name=name, trainable=trainable, **kwargs)
    self.axis = -1
    self.center = center
    self.scale = scale
    self.epsilon = epsilon
    self.supports_masking = True

  def build(self, input_shape):
    assert isinstance(input_shape, list), type(input_shape)
    assert len(input_shape) == 2, len(input_shape)
    x_shape, z_shape = input_shape
    assert len(x_shape) == 4, x_shape
    assert len(z_shape) == 2, z_shape
    x_dim = x_shape[self.axis]
    z_dim = z_shape[-1]
    if x_dim is None:
      raise ValueError(
        'Axis ' + str(self.axis) + ' of '
        'input tensor should have a defined dimension '
        'but the layer received an input with shape ' +
        str(x_shape) + '.'
      )
    if self.scale:
      self.gamma = self.add_weight(
        shape=(x_dim,),
        name='gamma',
        initializer='ones',
      )
      self.z_gamma = self.add_weight(
        shape=(z_dim, x_dim),
        name='z_gamma',
        initializer='zeros',
      )
    if self.center:
      self.beta = self.add_weight(
        shape=(x_dim,),
        name='beta',
        initializer='zeros',
      )
      self.z_beta = self.add_weight(
        shape=(z_dim, x_dim),
        name='z_beta',
        initializer='zeros',
      )
    self.axes = list(range(len(x_shape)))
    self.axes.pop(self.axis)
    self.built = True

  def call(self, inputs, training=None):
    x, z = inputs
    ctx = tf.distribute.get_replica_context()
    mean, mean_sq = ctx.all_reduce(
      tf.distribute.ReduceOp.SUM,
      [tf.reduce_mean(
        t, axis=self.axes, keepdims=True
      ) / ctx.num_replicas_in_sync
      for t in (x, x**2)]
    )
    variance = mean_sq - mean ** 2
    if self.scale:
      gamma = self.gamma + tf.matmul(z, self.z_gamma)
      gamma = gamma[:, tf.newaxis, tf.newaxis]
    else:
      gamma = None
    if self.center:
      beta = self.beta + tf.matmul(z, self.z_beta)
      beta = beta[:, tf.newaxis, tf.newaxis]
    else:
      beta = None
    return tf.nn.batch_normalization(
      x,
      mean,
      variance,
      beta,
      gamma,
      self.epsilon)

  def compute_output_shape(self, input_shape):
    return input_shape[0]

  def get_config(self):
    return {
      'axis': self.axis,
      'epsilon': self.epsilon,
      'center': self.center,
      'scale': self.scale,
    }
