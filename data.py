from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import os

def get_image_data(dirs, image_size, batch_size, n_threads=8):
  samples = tf.random.uniform(minval=-1, maxval=1, dtype=tf.float32, shape=(batch_size, image_size, image_size, 3))
  print(samples)
  return samples
  def read_and_decode(filename):
    return tf.image.decode_image(tf.read_file(tf.print(filename)), channels=3)
  def keep(img):
    return tf.reduce_min(tf.shape(img)[:2]) >= image_size
  def resize(img):
    img = tf.image.resize_image_with_crop_or_pad(img, *(2*[
      tf.reduce_min(tf.shape(img)[:2])]))
    return tf.image.resize_images(img, 2*[image_size],
      method=tf.image.ResizeMethod.AREA, align_corners=True)
  samples = []
  if not type(dirs) in (tuple, list):
    dirs = [dirs]
  for path in dirs:
    files = tf.gfile.Glob([os.path.join(path, '*.'+ext) for ext in ['png']])
    nf = len(files)
    print('found {} files'.format(nf))
    if not nf:
      raise ValueError('must provide directory with pngs')
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.shuffle(nf, reshuffle_each_iteration=True)
    ds = ds.repeat()
    ds = ds.map(read_and_decode, n_threads)
    ds = ds.filter(keep)
    ds = ds.map(resize, n_threads)
    ds = ds.batch(batch_size)
    ds = ds.map(preprocess_img)
    ds = ds.prefetch(n_threads)
    it = ds.make_one_shot_iterator()
    samp = it.get_next()
    samp.set_shape((batch_size, image_size, image_size, 3))
    samples.append(samp)
  if len(samples) == 1:
    return samples[0]
  return samples

def preprocess_img(img):
  return tf.cast(img, tf.float32) / 127.5 - 1

def postprocess_img(img):
  return tf.cast(tf.round(tf.clip_by_value(img * 127.5 + 127.5, 0, 255)), tf.uint8)
