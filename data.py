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
      push, pop, clear = q.put(tensors), q.get(), q.clear()
    tf.train.add_queue_runner(tf.train.QueueRunner(
      queue=q, enqueue_ops=[push]*n_threads, close_op=clear, cancel_op=clear))
    return pop
  return stage

@queue_on_gpu(memory_limit_gb=1, n_threads=1)
def get_image_data(dirs, crop_size, batch_size, n_threads=3):
  samples, sizes = [], []
  for path in dirs:
    files = []
    for ext in ['jpg', 'jpeg', 'png', 'bmp']:
      for ext in [ext.lower(), ext.upper()]:
        for depth in '*.', '*/*.':
          files += glob.glob(os.path.join(path, depth+ext))
    filenames = tf.train.string_input_producer(files, shuffle=True)
    img_bytes = tf.WholeFileReader().read(filenames)[1]
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    sizes.append(tf.shape(img)[:2])
    img = preprocess_img(img)
    img = tf.image.resize_image_with_crop_or_pad(img, *[
      tf.maximum(crop_size, s) for s in tf.unstack(tf.shape(img))[:2]])
    img = tf.random_crop(img, [crop_size, crop_size, 3])
    samples.append(img)
  keep = tf.cast(tf.logical_and(*[tf.greater_equal(tf.reduce_min(
    s), crop_size) for s in sizes]), tf.float32)
  return tf.contrib.training.rejection_sample(
    samples, lambda x: keep, batch_size,
    prebatch_capacity=n_threads*batch_size,
    queue_threads=n_threads)

def preprocess_img(img):
  return tf.cast(img, tf.float32) / 127.5 - 1

def postprocess_img(img):
  return tf.cast(tf.round(tf.clip_by_value(img * 127.5 + 127.5, 0, 255)), tf.uint8)
