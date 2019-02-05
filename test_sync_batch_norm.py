import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from sync_batch_norm import SyncBatchNorm

def make_bn_model():
  x = Input((64, 64, 64, 1))
  y = SyncBatchNorm()(x)
  return Model(inputs=x, outputs=y)
