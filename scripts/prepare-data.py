import logging
import argparse

from biggan.data import create_dataset


def parse_args():
    """
    Returns a dictionary of arguments parsed from the command
    line for `create_dataset` function
    """
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "data_dir",
        type=str,
        help="directory containing training PNGs and/or JPGs",
    )
    p.add_argument(
        "output_npz",
        type=str,
        help=".npz file in which to save processed images and labels",
    )
    p.add_argument(
        "-is",
        "--image_size",
        type=int,
        default=256,
        help="size of downsampled images",
    )
    return vars(p.parse_args())


def main(args=None):
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)
    create_dataset(**(args or parse_args()))

if __name__ == "__main__":
    main()
