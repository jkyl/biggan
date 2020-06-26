import argparse

import tensorflow as tf

from biggan import BigGAN
from biggan import get_train_data
from biggan import get_strategy


def build_model(
    *,
    channels: int,
    num_classes: int,
    checkpoint: str = None,
):
    """
    Builds the model within a multi-device context.
    """
    with get_strategy().scope():
        model = BigGAN(channels=channels, num_classes=num_classes)
        model.compile()
        if checkpoint is not None:
            model.load_weights(checkpoint)
    return model


def train_model(
    *,
    model: BigGAN,
    data: tf.data.Dataset,
    model_dir: str,
    num_steps: int,
    log_every: int,
    initial_step: int = 0,
):
    """
    Train the built model on a dataset.
    """
    # Create the saving and logging callbacks.
    callbacks = model.create_callbacks(model_dir)

    # Set the global batch size for distributed training.
    model.global_batch_size = data.element_spec[0].shape[0]

    # Fit the model to the data, calling the callbacks every `log_every` steps.
    model.fit(
        data,
        callbacks=callbacks,
        epochs=num_steps//log_every,
        steps_per_epoch=log_every,
        initial_epoch=(initial_step or 0) // log_every,
    )
    # Return the trained model.
    return model


def run(
    *,
    channels: int,
    data_file: str,
    model_dir: str,
    batch_size: int,
    num_steps: int,
    log_every: int,
):
    """
    Build a model, build a dataset, then train the model on the dataset.
    """

    # Create the dataset object from the .npz file.
    data = get_train_data(data_file=data_file, batch_size=batch_size)

    # Check if a checkpoint exists.
    checkpoint = tf.train.latest_checkpoint(model_dir)

    # Build the model.
    model = build_model(channels=channels, num_classes=data.num_classes, checkpoint=checkpoint)

    # Train the model on the dataset.
    return train_model(
        model=model,
        data=data,
        model_dir=model_dir,
        num_steps=num_steps,
        log_every=log_every,
        initial_step=model.G_adam.iterations,
    )


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
        "-ns",
        dest="num_steps",
        type=int,
        default=1_000_000,
        help="total number of training iterations",
    )
    p.add_argument(
        "-le",
        dest="log_every",
        type=int,
        default=100,
        help="interval of training steps at which to log output and save checkpoints",
    )
    return p.parse_args()


def main(args=None):
    return run(**vars(args or parse_arguments()))


if __name__ == "__main__":
    main()
