from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import argparse
import logging
import glob
import cv2
import os

from tensorflow.python.client import device_lib


def get_gpus():
  '''Returns the identities of all available GPUs
  '''
  return [
    d.name for d in 
    device_lib.list_local_devices() 
    if d.device_type == 'GPU']

def get_strategy():
  '''Returns a mirrored strategy over all available GPUs,
  or falls back to CPU if no GPUs available
  ''' 
  return tf.distribute.MirroredStrategy(
    devices=get_gpus() or ['/CPU:0'])

def preprocess(img):
  '''Casts a tensor's type to half-precision float, 
  then scales its values to the range [-1, 1]
  '''
  return tf.cast(img, tf.float16) / 127.5 - 1

def postprocess(img):
  '''Scales a tensor's values to the range [0, 255], 
  then casts its type to unsigned 8-bit integer
  '''
  return tf.cast(tf.clip_by_value(img * 127.5 + 127.5, 0, 255), tf.uint8)

def get_train_data(params, n_threads=8, cache=True):
  '''Creates a training data pipeline that samples batches 
  from an array of pre-cropped image data in the provided 
  .npy file, caching it in memory unless otherwise provided
  '''
  npy_file = params['data_file']
  batch_size = params['batch_size']
  n_gpus = len(get_gpus())
  if not n_gpus:
    pass
  elif batch_size % n_gpus:
    raise ValueError(
      'batch size ({}) is not evenly divisible by number of GPUs ({})'
      .format(batch_size, n_gpus))
  else:
    batch_size //= n_gpus
  data = np.load(npy_file, mmap_mode=None if cache else 'r')
  n, h, w, c = data.shape
  def gen():
    while True:
      yield data[np.random.randint(n, size=batch_size)]
  ds = tf.data.Dataset.from_generator(gen, tf.uint8, (batch_size, h, w, c))
  ds = ds.map(preprocess, n_threads)
  ds = ds.prefetch(n_threads)
  return ds

def _glob_image_files(data_dir): 
  '''Searches the given directory and its subdirectories for 
  JPEG and PNG files, and returns a list of their filenames
  '''
  files = []
  for depth in range(2):
    for ext in ('jpg', 'jpeg', 'png'):
      for case in (ext.lower(), ext.upper()):
        pattern = '/'.join('*' * (depth + 1)) + '.' + case
        files += glob.glob(os.path.join(data_dir, pattern))
  return files

class ImageTooSmall(Exception):
  pass

def _load_crop_resize_img(filename, image_size):
  '''Loads an image from disk, crops into a square along 
  its major axis, then downsamples to the given size
  '''
  # load the image and get its size
  image = cv2.imread(filename)
  size = image.shape[:2]
  min_ax = np.argmin(size)
  
  # make sure the image is at least as big as the final size
  if size[min_ax] < image_size:
    raise ImageTooSmall(size)

  # construct a crop
  crop = ()
  for j, dim in enumerate(size):
    if j == min_ax:
      start = stop = None
    else:
      start = (size[j] - size[min_ax]) // 2
      stop = start + image_size
    crop += (slice(start, stop, None),)
  crop += (slice(None, None, -1),)

  # apply the crop and resize
  return cv2.resize(
    image[crop], 
    (image_size, image_size),
    interpolation=cv2.INTER_AREA)

def _create_dataset(data_dir, output_npy, image_size=256):
  '''Loads PNG and JPEG image files within possibly nested 
  directories, crops them, downsamples them to the target size,
  puts them all into a single numpy array, then writes to disk
  '''
  # get the image filenames
  files = _glob_image_files(data_dir)
  num_files = len(files)
  if num_files == 0:
    raise IOError('no image files found in "{}"'.format(data_dir))
  
  # allocate an array in memory to store all the images
  arr = np.zeros((num_files, image_size, image_size, 3), dtype=np.uint8)
  i = 0

  # try to process every file
  for filename in files:
    try:
      arr[i] = _load_crop_resize_img(filename, image_size)
    
    # log non-fatal exceptions
    except Exception as err:
      if type(err) in (KeyboardInterrupt, SystemExit):
        raise err
      logging.error('{}: {} on file {}'
        .format(type(err).__name__, err, filename))

    else: # increment the counter
      i += 1

  # save the array to .npy file
  np.save(output_npy, arr[:i])

def _parse_args():
  '''Returns a dictionary of arguments parsed from the command
  line for `_create_dataset` function
  '''
  p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument('data_dir', type=str,
    help='directory containing training PNGs and/or JPGs')
  p.add_argument('output_npy', type=str,
    help='.npz file in which to save preprocessed images')
  p.add_argument('-is', '--image_size', type=int, default=128,
    help='size of downsampled images')
  return vars(p.parse_args())

if __name__ == '__main__':
  logging.basicConfig(
    format='%(levelname)s:%(message)s', 
    level=logging.DEBUG)
  _create_dataset(**_parse_args())

