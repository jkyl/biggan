#!/usr/bin/env python3

"""
Trains a BigGAN.
"""

__author__ = "Jon Kyl"


import tensorflow as tf
import biggan


cfg = biggan.config.base


def _run(args):
    """
    Builds a model, builds a dataset, then trains the model on the dataset.
    """

    # Create a dataset object from tfrecord files.
    dataset = biggan.get_tfrecord_dataset(
        image_size=args.image_size,
        tfrecord_path=args.tfrecord_path,
        batch_size=args.batch_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
        do_cache=args.do_cache,
    )
    # Get the latest checkpoint.
    checkpoint = tf.train.latest_checkpoint(args.model_path)
    if checkpoint is None:
        initial_epoch = 0
    else:
        initial_epoch = int(checkpoint.split("ckpt_")[-1].split(".")[0])

    # Build the model.
    model = biggan.build_model(
        image_size=args.image_size,
        channels=args.channels,
        num_classes=lambda: next(iter(dataset.take(1)))[1].shape[1],
        latent_dim=args.latent_dim,
        checkpoint=checkpoint,
        G_learning_rate=args.G_learning_rate,
        D_learning_rate=args.D_learning_rate,
        G_beta_1=args.G_beta_1,
        D_beta_1=args.D_beta_1,
        G_beta_2=args.G_beta_2,
        D_beta_2=args.D_beta_2,
        use_tpu=args.use_tpu,
        momentum=args.momentum,
        epsilon=args.epsilon,
        num_D_updates=args.num_D_updates,
    )
    # Train the model on the dataset.
    biggan.train_model(
        model=model,
        dataset=dataset,
        model_path=args.model_path,
        num_epochs=args.num_epochs,
        log_every=args.log_every,
        initial_epoch=initial_epoch,
    )


def main():
    return _run(cfg.parse_args())


if __name__ == "__main__":
    main()
