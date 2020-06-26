import tensorflow as tf
import biggan


def run(
    *,
    channels: int,
    data_file: str,
    model_dir: str,
    batch_size: int,
    num_steps: int,
    log_every: int,
    G_learning_rate: float,
    G_beta_1: float,
    G_beta_2: float,
    D_learning_rate: float,
    D_beta_1: float,
    D_beta_2: float,
):
    """
    Build a model, build a dataset, then train the model on the dataset.
    """

    # Create the dataset object from the .npz file.
    data = biggan.get_train_data(data_file=data_file, batch_size=batch_size)

    # Check if a checkpoint exists.
    checkpoint = tf.train.latest_checkpoint(model_dir)

    # Build the model.
    model = biggan.build_model(
        channels=channels,
        num_classes=data.num_classes,
        checkpoint=checkpoint,
        G_learning_rate=G_learning_rate,
        G_beta_1=G_beta_1,
        G_beta_2=G_beta_2,
        D_learning_rate=D_learning_rate,
        D_beta_1=D_beta_1,
        D_beta_2=D_beta_2,
    )

    # Train the model on the dataset.
    return biggan.train_model(
        model=model,
        data=data,
        model_dir=model_dir,
        num_steps=num_steps,
        log_every=log_every,
        initial_step=model.G_adam.iterations,
    )


def main(config=biggan.config.train):
    return run(**vars(config.parse_args()))


if __name__ == "__main__":
    main()
