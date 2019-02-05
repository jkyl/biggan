import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from sync_batch_norm import SyncBatchNorm

def make_bn_model():
  x = Input((64, 64, 1))
  y = SyncBatchNorm()(x)
  return Model(inputs=x, outputs=y)

if __name__ == '__main__':
  model = make_bn_model()
  model.predict(np.ones((1, 64, 64, 1)))