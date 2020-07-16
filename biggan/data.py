import tensorflow as tf
import numpy as np
import os

from .config import base as cfg


def preprocess_image(img):
    """
    Casts a tensor's type to from uint8 to float32,
    then scales its values to the range [-1, 1].
    """
    return tf.cast(img, tf.float32) / 127.5 - 1.


def postprocess_image(img):
    """
    Scales a tensor's values to the range [0, 255],
    then casts its type to unsigned 8-bit integer
    """
    return tf.cast(tf.round(tf.clip_by_value(img * 127.5 + 127.5, 0, 255)), tf.uint8)


def get_preprocessing_pipeline(data_path, image_size):
    """
    Streams resized images and their class labels from disk to a tf.data.Dataset.
    """

    def get_files_and_onehot_labels():
        files = tf.io.gfile.glob([
            os.path.join(data_path, f"*/*.{ext}")
            for ext in ("jpg", "jpeg", "png")
            for ext in (ext, ext.upper())])
        np.random.shuffle(files)
        classes = [
            os.path.basename(os.path.dirname(path))
            for path in files]
        unique_classes = np.unique(classes)
        onehot_labels = [unique_classes == c for c in classes]
        return np.array(files), np.array(onehot_labels, dtype=np.float32)

    def load(image_file):
        image_bytes = tf.io.read_file(image_file)
        return tf.image.decode_image(image_bytes, channels=3)

    def is_large_enough(image):
        return tf.reduce_min(tf.shape(image)[:2]) >= image_size

    def crop_and_resize(image):
        short_side = tf.reduce_min(tf.shape(image)[:2])
        image = tf.image.resize_with_crop_or_pad(image, short_side, short_side)
        image = tf.image.resize(image, (image_size, image_size), method=tf.image.ResizeMethod.AREA)
        image = tf.cast(tf.round(image), tf.uint8)
        image.set_shape((image_size, image_size, 3))
        return image

    # Get a dataset of files and their class labels.
    dataset = tf.data.Dataset.from_tensor_slices(get_files_and_onehot_labels())

    # Map to dataset of images and labels.
    dataset = dataset.map(lambda path, label: (load(path), label))

    # Ignore image decoding errors.
    dataset = dataset.apply(tf.data.experimental.ignore_errors())

    # Ignore images that are too small.
    dataset = dataset.filter(lambda image, label: is_large_enough(image))

    # Crop and resize the images.
    dataset = dataset.map(lambda image, label: (crop_and_resize(image), label))

    # Return the pipeline.
    return dataset


def serialize_to_tfrecords(
    *,
    input_path: str,
    output_path: str,
    image_size: int,
    num_examples_per_shard: int,
):
    """
    Serializes images in `input_path` along with their class labels
    to multiple tfrecord files.
    """

    def serialize(image, label):
        """
        Returns a string-serialized tf.train.Example containing
        a single image and class label.
        """
        return tf.train.Example(features=tf.train.Features(feature=dict(
            image=tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.numpy().tobytes()])),
            label=tf.train.Feature(float_list=tf.train.FloatList(value=label.numpy().tolist()))))
        ).SerializeToString()

    def tf_serialize(image, label):
        """
        Wraps the `serialize` function with `tf.py_function` so that
        it can be invoked in a data pipeline.
        """
        tf_string = tf.py_function(
            serialize,
            (image, label),
            tf.string
        )
        return tf.reshape(tf_string, ())

    # Create the preprocessing pipeline.
    dataset = get_preprocessing_pipeline(input_path, image_size=image_size)

    # Serialize the preprocessed data.
    dataset = dataset.map(tf_serialize)

    # Shard the dataset.
    dataset = dataset.batch(num_examples_per_shard).prefetch(2)

    # Make the output directory if it does not exist.
    if not tf.io.gfile.exists(output_path):
        tf.io.gfile.makedirs(output_path)

    # Write each shard to its own tfrecord file.
    for i, elem in enumerate(iter(dataset)):
        filename = os.path.join(output_path, f"shard.{str(i).zfill(4)}.tfrecord.gz")
        with tf.io.TFRecordWriter(filename, options="GZIP") as writer:
            for proto in elem:
                writer.write(proto.numpy())


def get_tfrecord_dataset(
    tfrecord_path: str,
    *,
    image_size: int,
    batch_size: int = cfg.defaults.batch_size,
    shuffle_buffer_size: int = cfg.defaults.shuffle_buffer_size,
    do_cache: bool = cfg.defaults.do_cache,
):
    """
    Loads all tfrecord files stored in `tfrecord_path` into a dataset.
    """

    def parse_example(proto):
        """
        Parses a single training example back into tensors of the right type and shape.
        """
        parsed = tf.io.parse_single_example(
            serialized=proto,
            features=dict(
                image=tf.io.FixedLenFeature([], tf.string),
                label=tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            )
        )
        image, label = parsed["image"], parsed["label"]
        image = tf.io.decode_raw(image, out_type=np.uint8)
        image = tf.reshape(image, (image_size, image_size, 3))
        return image, label

    # Look for tfrecord files in `tfrecord_path`.
    tfrecord_files = tf.io.gfile.glob(os.path.join(tfrecord_path, "*.tfrecord.gz"))

    # Create the tfrecord dataset.
    dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type="GZIP")

    # Parse the serialized examples into tensors.
    dataset = dataset.map(parse_example)

    if do_cache:
        # Cache the entire dataset.
        dataset = dataset.cache()

    # Shuffle the dataset in chunks of `shuffle_buffer_size`.
    dataset = dataset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)

    # Combine the tensors into batches.
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Convert the images and labels to floating-point.
    dataset = dataset.map(lambda image, label: (preprocess_image(image), label))

    # Prefetch batches with an automatically determined number of threads.
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Return the dataset for training.
    return dataset
