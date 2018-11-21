import os, glob, signal, tqdm, cv2, PIL
import tensorflow as tf
import numpy as np

class timeout:
  def __init__(self, seconds=1, error_message='Timeout'):
    self.seconds = seconds
    self.error_message = error_message
  def handle_timeout(self, signum, frame):
    raise TimeoutError(self.error_message)
  def __enter__(self):
    signal.signal(signal.SIGALRM, self.handle_timeout)
    signal.alarm(self.seconds)
  def __exit__(self, type, value, traceback):
    signal.alarm(0)

def parameterized(decorator):
  def wrap(*args, **kwargs):
    def meta(f):
      return decorator(f, *args, **kwargs)
    return meta
  return wrap

@parameterized
def tf_func(f, *dtypes):
  def wrapped(*args):
    return tf.py_func(f, args, dtypes)
  return wrapped

@parameterized
def queue_on_gpu(data_function, _sentinel=None, memory_limit_gb=None, n_threads=None):
  if _sentinel or not (memory_limit_gb and n_threads):
    raise ValueError('must specify `memory_limit_gb` and `n_threads` as kwargs to `queue_on_gpu`')
  def stage(*args, **kwargs):
    with tf.device('/cpu:0'):
      tensors = data_function(*args, **kwargs)
    with tf.device('/gpu:0'):
      dtypes = [t.dtype for t in tensors]
      shapes = [t.shape for t in tensors]
      q = tf.contrib.staging.StagingArea(
        dtypes, shapes=shapes, memory_limit=1e9*memory_limit_gb)
      push, pop, clear, size = q.put(tensors), q.get(), q.clear(), q.size()
    tf.summary.scalar('gpu_queue_size', size)
    tf.train.add_queue_runner(tf.train.QueueRunner(
      queue=q, enqueue_ops=[push]*n_threads, close_op=clear, cancel_op=clear))
    return pop
  return stage

@queue_on_gpu(memory_limit_gb=0.2, n_threads=1)
def get_image_data(dirs, image_size, batch_size, mode='random_crop', n_threads=8):
  @tf_func(tf.uint8)
  def read_and_decode(filename):
    return cv2.imread(filename.decode('utf-8'))
  def keep(img):
    return tf.reduce_min(tf.shape(img)[:2]) >= image_size
  def random_crop(img):
    return tf.random_crop(img, [image_size, image_size, 3])
  def resize(img):
    img = tf.image.resize_image_with_crop_or_pad(img, *([
      tf.reduce_min(tf.shape(img)[:2])]*2))
    return tf.image.resize_images(img, [image_size]*2,
      method=tf.image.ResizeMethod.AREA, align_corners=True)
  samples = []
  for path in dirs:
    files = []
    for ext in ['jpg', 'jpeg', 'png']:
      for ext in [ext.lower(), ext.upper()]:
        for depth in '*.', '*/*.':
          files += glob.glob(os.path.join(path, depth+ext))
    nf = len(files)
    print('found {} files'.format(nf))
    if not nf:
      raise ValueError('must provide directory with jpegs or pngs')
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.shuffle(nf, reshuffle_each_iteration=True)
    ds = ds.repeat()
    ds = ds.map(read_and_decode, n_threads)
    ds = ds.filter(keep)
    ds = ds.map(resize if mode == 'resize' else random_crop, n_threads)
    ds = ds.batch(batch_size)
    ds = ds.map(preprocess_img)
    ds = ds.prefetch(n_threads)
    it = ds.make_one_shot_iterator()
    samples.append(it.get_next())
  return samples

def preprocess_img(img):
  return tf.cast(img[...,::-1], tf.float32) / 127.5 - 1

def postprocess_img(img):
  return tf.cast(tf.round(tf.clip_by_value(img * 127.5 + 127.5, 0, 255)), tf.uint8)