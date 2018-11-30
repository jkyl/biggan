from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import os

def get_image_data(dirs, image_size, batch_size, n_threads=8):
  def read_and_decode(filename):
    x = tf.image.decode_image(tf.read_file(filename), channels=3)
    return x
  def keep(img):
    return tf.reduce_min(tf.shape(img)[:2]) >= image_size
  def resize(img):
    img = tf.image.resize_image_with_crop_or_pad(img, *([
      tf.reduce_min(tf.shape(img)[:2])]*2))
    return tf.image.resize_images(img, [image_size]*2,
      method=tf.image.ResizeMethod.AREA, align_corners=True)
  samples = []
  if not type(dirs) in (tuple, list):
    dirs = [dirs]
  for path in dirs:
    files = []
    for ext in ['jpg', 'jpeg', 'png']:
      for ext in [ext.lower(), ext.upper()]:
        for depth in '*.', '*/*.':
          files += tf.gfile.Glob(os.path.join(path, depth+ext))
    nf = len(files)
    print('found {} files'.format(nf))
    if not nf:
      raise ValueError('must provide directory with jpegs or pngs')
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
  return tf.cast(img[..., ::-1], tf.float32) / 127.5 - 1

def postprocess_img(img):
  return tf.cast(tf.round(tf.clip_by_value(img * 127.5 + 127.5, 0, 255)), tf.uint8)
