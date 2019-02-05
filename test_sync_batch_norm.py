import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from sync_batch_norm import SyncBatchNorm

def make_bn_model():
  x = Input((64, 64, 1))
  y = SyncBatchNorm()(x)
  return Model(inputs=x, outputs=y)

def test():
  model = make_bn_model()
  inputs = np.random.normal(size=(4, 64, 64, 1))
  inputs *= 2
  inputs += 1
  outputs = model.predict(inputs)
  return outputs