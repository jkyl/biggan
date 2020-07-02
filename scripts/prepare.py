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
        "input_path",
        type=str,
        help="directory containing training PNGs and/or JPEGs, separated by class into subdirectories",
    )
    p.add_argument(
        "output_path",
        type=str,
        help="path in which to save processed images and labels as gzipped tfrecords",
    )
    return vars(p.parse_args())


def main(args=None):
    serialize_to_tfrecords(**(args or parse_args()))

if __name__ == "__main__":
    main()
