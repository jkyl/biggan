#!/usr/bin/env python3

"""
Trains a BigGAN.
"""

import tensorflow as tf
import biggan


cfg = biggan.config.base


def run(
    *,
    tfrecord_path: str,
    model_path: str,
    channels: int = cfg.defaults.channels,
    batch_size: int = cfg.defaults.batch_size,
    num_epochs: int = cfg.defaults.num_epochs,
    log_every: int = cfg.defaults.log_every,
    G_learning_rate: float = cfg.defaults.G_learning_rate,
    D_learning_rate: float = cfg.defaults.D_learning_rate,
    G_beta_1: float = cfg.defaults.G_beta_1,
    D_beta_1: float = cfg.defaults.D_beta_1,
    G_beta_2: float = cfg.defaults.G_beta_2,
    D_beta_2: float = cfg.defaults.D_beta_2,
    shuffle_buffer_size: int = cfg.defaults.shuffle_buffer_size,
    do_cache: bool = cfg.defaults.do_cache,
    latent_dim: int = cfg.defaults.latent_dim,
    **unused_kwargs,
):
    """
    Builds a model, builds a dataset, then trains the model on the dataset.
    """

    # Delete the unused keyword arguments.
    del unused_kwargs

    # Create the dataset object from the NPZ file.
    data = biggan.get_tfrecord_dataset(
        tfrecord_path=tfrecord_path,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        do_cache=do_cache,
    )
    # Build the model.
    model = biggan.build_model(
        channels=channels,
        num_classes=next(iter(data.take(1)))[1].shape[1],
        latent_dim=latent_dim,
        checkpoint=tf.train.latest_checkpoint(model_path),
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
        model_path=model_path,
        num_epochs=num_epochs,
        log_every=log_every,
    )


def main():
    return run(**vars(cfg.parse_args()))


if __name__ == "__main__":
    main()
