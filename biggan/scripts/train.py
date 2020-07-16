#!/usr/bin/env python3

"""
Trains a BigGAN.
"""

__author__ = "Jon Kyl"


import tensorflow as tf
import biggan


cfg = biggan.config.base


def run(
    *,
    tfrecord_path: str,
    model_path: str,
    image_size: int,
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
    use_tpu: bool = cfg.defaults.use_tpu,
    momentum: float = cfg.defaults.momentum,
    epsilon: float = cfg.defaults.momentum,
    num_D_updates: int = cfg.defaults.num_D_updates,
    **unused_kwargs,
):
    """
    Builds a model, builds a dataset, then trains the model on the dataset.
    """

    # Delete the unused keyword arguments.
    del unused_kwargs

    # Create a dataset object from tfrecord files.
    dataset = biggan.get_tfrecord_dataset(
        image_size=image_size,
        tfrecord_path=tfrecord_path,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        do_cache=do_cache,
    )
    # Get the latest checkpoint.
    checkpoint = tf.train.latest_checkpoint(model_path)
    if checkpoint is None:
        initial_epoch = 0
    else:
        initial_epoch = int(checkpoint.split("ckpt_")[-1].split(".")[0])

    # Build the model.
    model = biggan.build_model(
        image_size=image_size,
        channels=channels,
        num_classes=lambda: next(iter(dataset.take(1)))[1].shape[1],
        latent_dim=latent_dim,
        checkpoint=checkpoint,
        G_learning_rate=G_learning_rate,
        D_learning_rate=D_learning_rate,
        G_beta_1=G_beta_1,
        D_beta_1=D_beta_1,
        G_beta_2=G_beta_2,
        D_beta_2=D_beta_2,
        use_tpu=use_tpu,
        momentum=momentum,
        epsilon=epsilon,
        num_D_updates=num_D_updates,
    )
    # Train the model on the dataset.
    biggan.train_model(
        model=model,
        dataset=dataset,
        model_path=model_path,
        num_epochs=num_epochs,
        log_every=log_every,
        initial_epoch=initial_epoch,
    )


def main():
    return run(**vars(cfg.parse_args()))


if __name__ == "__main__":
    main()
