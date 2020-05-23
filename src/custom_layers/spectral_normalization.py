from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.python.keras.utils import tf_utils

def _spectrally_normalize_weight(
    weight,
    right_singular_vector,
    training=None,
  ):
  if training is None:
    training = K.learning_phase()

  def _l2normalize(v):
    return v / (K.sum(v ** 2) ** 0.5 + 1e-8)

  def power_iteration(W, u):
    _u = u
    _v = _l2normalize(K.dot(_u, K.transpose(W)))
    _u = _l2normalize(K.dot(_v, W))
    return tf.stop_gradient(_u), tf.stop_gradient(_v)

  W_shape = weight.shape.as_list()
  W_reshaped = K.reshape(weight, [-1, W_shape[-1]])
  _u, _v = power_iteration(W_reshaped, right_singular_vector)
  sigma = K.dot(_v, W_reshaped)
  sigma = K.dot(sigma, K.transpose(_u))
  W_bar = W_reshaped / sigma

  def assign_update():
    with tf.control_dependencies([right_singular_vector.assign(_u)]):
      return K.reshape(W_bar, W_shape)

  W_bar = tf_utils.smart_cond(
    training,
    assign_update,
    lambda: K.reshape(W_bar, W_shape))

  return W_bar

def _create_right_singular_vector(layer):
  return layer.add_weight(
    shape=tuple([1, layer.kernel.shape.as_list()[-1]]),
    initializer=initializers.RandomNormal(0, 1),
    name='sn',
    trainable=False,
    synchronization=tf.VariableSynchronization.ON_READ,
    aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
  )

class SpectralConv2D(Conv2D):
  def build(self, input_shape):
    super().build(input_shape)
    self.u = _create_right_singular_vector(self)

  def call(self, inputs, training=None):
    W_bar = _spectrally_normalize_weight(
      self.kernel,
      self.u,
      training=training,
    )
    outputs = K.conv2d(
      inputs,
      W_bar,
      strides=self.strides,
      padding=self.padding,
      data_format=self.data_format,
      dilation_rate=self.dilation_rate)
    if self.use_bias:
      outputs = K.bias_add(
        outputs,
        self.bias,
        data_format=self.data_format)
    if self.activation is not None:
      return self.activation(outputs)
    return outputs

class SpectralDense(Dense):
  def build(self, input_shape):
    super().build(input_shape)
    self.u = _create_right_singular_vector(self)

  def call(self, inputs, training=None):
    W_bar = _spectrally_normalize_weight(
      self.kernel,
      self.u,
      training=training,
    )
    output = K.dot(inputs, W_bar)
    if self.use_bias:
      output = K.bias_add(output, self.bias, data_format='channels_last')
    if self.activation is not None:
      output = self.activation(output)
    return output
