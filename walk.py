import tqdm
import numpy as np
from keras.models import load_model
from scipy.interpolate import interp1d
from skvideo.io import vwrite

def truncated_normal(n, thresh=np.inf):
  z = np.random.normal(size=n)
  m = np.abs(z) > thresh
  n = m.sum()
  if n:
    z[m] = truncated_normal(n, thresh)
  return z

def make_walk(n_total, truncations, z_dim=128):
  landmarks = np.array([truncated_normal(z_dim, t) for t in truncations])
  output = np.zeros((n_total, z_dim))
  xs = np.linspace(0, len(landmarks)-1, n_total)
  for col in range(z_dim):
    f = interp1d(np.arange(len(landmarks)), landmarks[:, col], kind='cubic', axis=0)
    output[:, col] = f(xs)
  return output

def predict_walk(model, n_total, truncations, batch_size=16):
  zs = make_walk(n_total, truncations, model.input_shape[-1])
  n_batches = n_total / float(batch_size)
  output = np.empty((n_total,)+model.output_shape[1:], dtype=np.float32)
  if int(n_batches) != n_batches:
    n_batches = int(np.ceil(n_batches))
  else:
    n_batches = int(n_batches)
  for i in tqdm.trange(n_batches):
    output[i*batch_size:(i+1)*batch_size] = model.predict(
      zs[i*batch_size:(i+1)*batch_size], batch_size=batch_size)
  return output

def main(h5_filename, mp4_filename):
  model = load_model(h5_filename)
  n_total = 600
  truncations = np.linspace(1, 100, 8)
  imgs = predict_walk(model, n_total, truncations)
  imgs = np.around(imgs * 127.5 + 127.5).astype(np.uint8)
  vwrite(mp4_filename, imgs)
