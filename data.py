from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import argparse
import glob
import sys
import os

def get_gpus():
  from tensorflow.python.client import device_lib
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']

def preprocess_img(img):
  return tf.cast(img, tf.float16) / 127.5 - 1

def postprocess_img(img):
  return tf.cast(tf.round(tf.clip_by_value(img * 127.5 + 127.5, 0, 255)), tf.uint8)

def get_train_data(npz_file, batch_size, n_threads=8):
  print('loading dataset into memory...')
  data = np.load(npz_file)['data']
  n, h, w, c = data.shape
  print('done!')
  batch_size //= len(get_gpus())
  def gen():
    while 1:
      yield data[np.random.randint(n, size=batch_size)]
  ds = tf.data.Dataset.from_generator(gen, tf.uint8, (batch_size, h, w, c))
  ds = ds.map(preprocess_img, n_threads)
  ds = ds.prefetch(n_threads)
  print(ds)
  assert False
  return ds

def main(args):
  import tqdm
  import cv2
  files = []
  for ext in ('jpg', 'jpeg', 'png'):
    for case in (ext.lower(), ext.upper()):
      for depth in ('*.', '*/*.'):
        files += glob.glob(os.path.join(args.data_dir, depth + case))
  arr = np.zeros((len(files), args.image_size, args.image_size, 3), dtype=np.uint8)
  i = 0
  for f in tqdm.tqdm(files):
    image = cv2.imread(f)
    size = image.shape[:2]
    min_dim = np.argmin(size)
    if size[min_dim] >= args.image_size:
      try:
        max_dim = int(not min_dim)
        crop_start = (size[max_dim] - size[min_dim]) // 2
        crop_stop = crop_start + size[min_dim]
        crop = [slice(crop_start, crop_stop, None) if i == max_dim
          else slice(None, None, None) for i in (0, 1)]
        crop += [slice(None, None, -1)]
        image = image[crop]
        image = cv2.resize(image, (args.image_size, args.image_size),
          interpolation=cv2.INTER_AREA)
        arr[i] = image
        i += 1
      except:
        continue
  np.savez(args.output_npz, data=arr[:i])

if __name__ == '__main__':
  p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument('data_dir', type=str,
    help='directory containing training PNGs and/or JPGs')
  p.add_argument('output_npz', type=str,
    help='.npz file in which to save preprocessed images')
  p.add_argument('-is', '--image_size', type=int, default=128,
    help='size of downsampled images')
  sys.exit(main(p.parse_args()))