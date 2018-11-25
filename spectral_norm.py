from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorflow.python.framework.ops import control_dependencies
from tensorflow.python.keras.engine import *
from tensorflow.python.keras.layers import Conv2D, Dense
from tensorflow.python.keras import initializers
from tensorflow.python.keras import backend as K

class ConvSN2D(Conv2D):
  def build(self, input_shape):
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape.as_list()[channel_axis]
    kernel_shape = self.kernel_size + (input_dim, self.filters)
    self.kernel = self.add_weight(shape=kernel_shape,
                                  initializer=self.kernel_initializer,
                                  name='kernel',
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
    if self.use_bias:
      self.bias = self.add_weight(shape=(self.filters,),
                                  initializer=self.bias_initializer,
                                  name='bias',
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint)
    else:
      self.bias = None
    self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                             initializer=initializers.RandomNormal(0, 1),
                             name='sn',
                             trainable=False)
    self.input_spec = InputSpec(ndim=self.rank + 2,
                                axes={channel_axis: input_dim})
    self.built = True

  def call(self, inputs):
    def _l2normalize(v, eps=1e-12):
      return v / (K.sum(v ** 2) ** 0.5 + eps)
    def power_iteration(W, u):
      _u = u
      _v = _l2normalize(K.dot(_u, K.transpose(W)))
      _u = _l2normalize(K.dot(_v, W))
      return _u, _v
    W_shape = self.kernel.shape.as_list()
    W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
    _u, _v = power_iteration(W_reshaped, self.u)
    sigma = K.dot(_v, W_reshaped)
    sigma = K.dot(sigma, K.transpose(_u))
    W_bar = W_reshaped / sigma
    if training in {0, False}:
      W_bar = K.reshape(W_bar, W_shape)
    else:
      with control_dependencies([self.u.assign(_u)]):
        W_bar = K.reshape(W_bar, W_shape)
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

class DenseSN(Dense):
  def build(self, input_shape):
    assert len(input_shape) >= 2
    input_dim = input_shape.as_list()[-1]
    self.kernel = self.add_weight(shape=(input_dim, self.units),
                                  initializer=self.kernel_initializer,
                                  name='kernel',
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
    if self.use_bias:
      self.bias = self.add_weight(shape=(self.units,),
                                  initializer=self.bias_initializer,
                                  name='bias',
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint)
    else:
      self.bias = None
    self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                             initializer=initializers.RandomNormal(0, 1),
                             name='sn',
                             trainable=False)
    self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
    self.built = True

  def call(self, inputs, training=None):
    def _l2normalize(v, eps=1e-12):
      return v / (K.sum(v ** 2) ** 0.5 + eps)
    def power_iteration(W, u):
      _u = u
      _v = _l2normalize(K.dot(_u, K.transpose(W)))
      _u = _l2normalize(K.dot(_v, W))
      return _u, _v
    W_shape = self.kernel.shape.as_list()
    W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
    _u, _v = power_iteration(W_reshaped, self.u)
    sigma = K.dot(_v, W_reshaped)
    sigma = K.dot(sigma, K.transpose(_u))
    W_bar = W_reshaped / sigma
    if training in {0, False}:
      W_bar = K.reshape(W_bar, W_shape)
    else:
      with control_dependencies([self.u.assign(_u)]):
        W_bar = K.reshape(W_bar, W_shape)
    output = K.dot(inputs, W_bar)
    if self.use_bias:
      output = K.bias_add(output, self.bias, data_format='channels_last')
    if self.activation is not None:
      output = self.activation(output)
    return output
