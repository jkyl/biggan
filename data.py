import os, glob
import tensorflow as tf

def parameterized(decorator):
  def wrap(*args, **kwargs):
    def meta(f):
      return decorator(f, *args, **kwargs)
    return meta
  return wrap

@parameterized
def queue_on_gpu(data_function, memory_limit_gb, n_threads):
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

@queue_on_gpu(memory_limit_gb=1, n_threads=4)
def get_image_data(dirs, crop_size, batch_size, n_threads=8):
  def read_and_decode(filename):
    return tf.image.decode_jpeg(tf.read_file(filename), channels=3, try_recover_truncated=True)
  def keep(img):
    return tf.reduce_min(tf.shape(img)[:2]) >= crop_size
  def random_crop(img):
    return tf.random_crop(img, [crop_size, crop_size, 3])
  samples = []
  for path in dirs:
    files = []
    for ext in ['jpg', 'jpeg', 'png', 'bmp']:
      for ext in [ext.lower(), ext.upper()]:
        for depth in '*.', '*/*.':
          files += glob.glob(os.path.join(path, depth+ext))
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.shuffle(len(files), reshuffle_each_iteration=True)
    ds = ds.repeat()
    ds = ds.map(read_and_decode, n_threads)
    ds = ds.filter(keep)
    ds = ds.map(random_crop, n_threads)
    ds = ds.batch(batch_size)
    ds = ds.map(preprocess_img)
    ds = ds.prefetch(n_threads)
    it = ds.make_one_shot_iterator()
    samples.append(it.get_next())
  return samples

def preprocess_img(img):
  return tf.cast(img, tf.float32) / 127.5 - 1

def postprocess_img(img):
  return tf.cast(tf.round(tf.clip_by_value(img * 127.5 + 127.5, 0, 255)), tf.uint8)
