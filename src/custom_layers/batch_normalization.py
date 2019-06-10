from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

class HyperBatchNorm(tf.keras.layers.Layer):
  """Cross-replica batch normalization layer"""
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
    input_shape, z_shape = input_shape
    dim = input_shape[self.axis]
    z_dim = z_shape[-1]
    if dim is None:
      raise ValueError(
        'Axis ' + str(self.axis) + ' of '
        'input tensor should have a defined dimension '
        'but the layer received an input with shape ' +
        str(input_shape) + '.'
      )
    shape = (dim,)
    if self.scale:
      self.gamma = self.add_weight(
        shape=shape,
        name='gamma',
        initializer='ones',
      )
      self.z_gamma = self.add_weight(
        shape=(z_dim, dim),
        name='z_gamma',
        initializer='zeros',
      )
    if self.center:
      self.beta = self.add_weight(
        shape=shape,
        name='beta',
        initializer='zeros',
      )
      self.z_beta = self.add_weight(
        shape=(z_dim, dim),
        name='z_beta',
        initializer='zeros',
      )
    self.axes = list(range(len(input_shape)))
    self.axes.pop(self.axis)
    self.built = True

  def call(self, inputs, training=None):
    x, z = inputs
    ctx = tf.distribute.get_replica_context()
    n = ctx.num_replicas_in_sync
    mean, mean_sq = ctx.all_reduce(
      tf.distribute.ReduceOp.SUM,
      [tf.reduce_mean(x, axis=self.axes, keepdims=True) / n,
       tf.reduce_mean(x**2, axis=self.axes, keepdims=True) / n])
    variance = mean_sq - mean ** 2
    reciprocal = tf.math.rsqrt(variance + self.epsilon)
    if self.scale:
      gamma = self.gamma + tf.matmul(z, self.z_gamma)
      reciprocal *= gamma[:, tf.newaxis, tf.newaxis, :]
    x *= reciprocal
    offset = -mean * reciprocal
    if self.center:
      beta = self.beta + tf.matmul(z, self.z_beta)
      offset += beta[:, tf.newaxis, tf.newaxis, :]
    x += offset
    return x

  def compute_output_shape(self, input_shape):
    return input_shape[0]

  def get_config(self):
    return {
      'axis': self.axis,
      'epsilon': self.epsilon,
      'center': self.center,
      'scale': self.scale,
    }
