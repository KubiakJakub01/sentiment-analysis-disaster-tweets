"""
Module with utility functions to fit custom transformers
"""

import tensorflow as tf

def fit_custom_transformer(
    model: tf.keras.models.Model,
    train_dataset: tf.data.Dataset,
    valid_dataset: tf.data.Dataset,
    params: dict,
    optimizer: tf.keras.optimizers.Optimizer,
    metrics: list,
    callbacks: list,
) -> None:
    """Fit the custom transformer model.

    Args:
        model (tf.keras.models.Model): Model to fit.
        train_dataset (tf.data.Dataset): Train dataset.
        valid_dataset (tf.data.Dataset): Valid dataset.
        params (dict): Dictionary containing the parameters.
        optimizer (tf.keras.optimizers.Optimizer): Optimizer to use.
        metrics (list): List of metrics to use.
        callbacks (list): List of callbacks to use.
    """

    # Compile the model
    model.compile(optimizer=optimizer, 
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=metrics)

    # Fit the model
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        callbacks=callbacks,
    )

    # Unfreeze the model
    for layer in model.layers:
        layer.trainable = True
    
    # Compile the model
    model.compile(optimizer=optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=metrics)
    
    # Fit the model
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        callbacks=callbacks,
    )
