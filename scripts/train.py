import argparse

from biggan import BigGAN
from biggan import get_train_data
from biggan import get_strategy


def build_model_and_train(
    *,
    model_dir: str,
    data_file: str,
    batch_size: int,
    channels: int,
    num_classes: int,
    num_steps: int = 1_000_000,
):
    """
    """
    # Create the dataset object from the .npz file.
    data = get_train_data(data_file=data_file, batch_size=batch_size)

    # Build the model within a multi-device context.
    with get_strategy().scope():
        model = BigGAN(channels=channels, num_classes=num_classes)
        model.compile(global_batch_size=batch_size)

    # Create the saving and logging callbacks.
    callbacks = model.create_callbacks(model_dir)

    # Fit the model to the data, calling the callbacks every 100 steps.
    model.fit(data, callbacks=callbacks, epochs=num_steps//100, steps_per_epoch=100)


def parse_arguments():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "data_file",
        type=str,
        help=".npy file containing prepared image data",
    )
    p.add_argument(
        "model_dir",
        type=str,
        help="directory in which to save checkpoints and logs",
    )
    p.add_argument(
        "-bs",
        dest="batch_size",
        type=int,
        default=64,
        help="total number of samples per gradient update",
    )
    p.add_argument(
        "-ch",
        dest="channels",
        type=int,
        default=48,
        help="greatest common factor of the number of channels in all layers",
    )
    p.add_argument(
        "-cl",
        dest="num_classes",
        type=int,
        default=27,
        help="number of image classes",
    )
    return p.parse_args()


def main(args=None):
    build_model_and_train(**vars(args or parse_arguments()))

if __name__ == "__main__":
    main()