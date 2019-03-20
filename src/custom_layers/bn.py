import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.engine.base_layer import InputSpec, Layer

class SyncBatchNorm(Layer):
  """Cross-replica batch normalization layer"""
  def __init__(self,
               center=True,
               scale=False,
               trainable=True,
               name=None,
               **kwargs):
    super(SyncBatchNorm, self).__init__(
        name=name, trainable=trainable, **kwargs)

    self.axis = -1
    self.center = center
    self.scale = scale
    self.supports_masking = True
    self.epsilon = 1e-3

  def build(self, input_shape):
    dim = input_shape[self.axis]
    if dim is None:
      raise ValueError('Axis ' + str(self.axis) + ' of '
                       'input tensor should have a defined dimension '
                       'but the layer received an input with shape ' +
                       str(input_shape) + '.')
    self.input_spec = InputSpec(ndim=len(input_shape),
                                axes={self.axis: dim})
    shape = (dim,)

    if self.scale:
      self.gamma = self.add_weight(shape=shape,
                                   name='gamma',
                                   initializer='ones',
                                   )
    else:
      self.gamma = None
    if self.center:
      self.beta = self.add_weight(shape=shape,
                                  name='beta',
                                  initializer='zeros',
                                  )
    else:
      self.beta = None
    self.built = True

  def call(self, x):
    ctx = tf.distribute.get_replica_context()
    n = ctx.num_replicas_in_sync
    mean, mean_sq = ctx.all_reduce(tf.distribute.ReduceOp.SUM, [
      K.mean(x, axis=0) / n, K.mean(x**2, axis=0) / n])
    variance = mean_sq - mean ** 2
    return tf.nn.batch_normalization(
      x,
      mean,
      variance,
      self.beta,
      self.gamma,
      self.epsilon)

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    return {
      'axis': self.axis,
      'epsilon': self.epsilon,
      'center': self.center,
      'scale': self.scale,
    }
