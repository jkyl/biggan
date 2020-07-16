import argparse

from biggan import serialize_to_tfrecords


def parse_args():
    """
    Returns a dictionary of arguments parsed from the command
    line for `create_dataset` function
    """
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input_path",
        required=True,
        type=str,
        help="directory containing training PNGs and/or JPEGs, separated by class into subdirectories",
    )
    p.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="path in which to save processed images and labels as gzipped tfrecords",
    )
    p.add_argument(
        "--image_size",
        required=True,
        type=int,
        choices=(128, 256, 512),
        help="sidelength of the images in pixels",
    )
    p.add_argument(
        "--num_examples_per_shard",
        type=int,
        default=2048,
        help="number of images and labels to put in each tfrecord file",
    )
    return vars(p.parse_args())


def main(args=None):
    serialize_to_tfrecords(**(args or parse_args()))

if __name__ == "__main__":
    main()
