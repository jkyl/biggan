#!/usr/bin/env python3

"""
Trains a BigGAN.
"""

import tensorflow as tf
import biggan


config = biggan.config.default


def run(
    *,
    data_file: str,
    model_dir: str,
    channels: int = config.defaults.channels,
    batch_size: int = config.defaults.batch_size,
    num_steps: int = config.defaults.num_steps,
    log_every: int = config.defaults.log_every,
    G_learning_rate: float = config.defaults.G_learning_rate,
    D_learning_rate: float = config.defaults.D_learning_rate,
    G_beta_1: float = config.defaults.G_beta_1,
    D_beta_1: float = config.defaults.D_beta_1,
    G_beta_2: float = config.defaults.G_beta_2,
    D_beta_2: float = config.defaults.D_beta_2,
):
    """
    Builds a model, builds a dataset, then trains the model on the dataset.
    """

    # Create the dataset object from the NPZ file.
    data = biggan.get_train_data(
        data_file=data_file,
        batch_size=batch_size,
    )
    # Build the model.
    model = biggan.build_model(
        channels=channels,
        num_classes=data.num_classes,
        checkpoint=tf.train.latest_checkpoint(model_dir),
        G_learning_rate=G_learning_rate,
        D_learning_rate=D_learning_rate,
        G_beta_1=G_beta_1,
        D_beta_1=D_beta_1,
        G_beta_2=G_beta_2,
        D_beta_2=D_beta_2,
    )
    # Train the model on the dataset.
    biggan.train_model(
        model=model,
        data=data,
        model_dir=model_dir,
        num_steps=num_steps,
        log_every=log_every,
        initial_step=model.G_adam.iterations,
    )


def main():
    return run(**vars(config.parse_args()))


if __name__ == "__main__":
    main()
