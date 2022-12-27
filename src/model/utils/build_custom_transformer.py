"""
Module with utility functions to build custom transformers 
based on the pre-trained models.
"""

import tensorflow as tf


def add_input_and_binary_output_layers(transformer, max_len):
    """Build a binary classifier based on the pre-trained transformer.

    Args:
        transformer (transformers.modeling_tf_utils.TFPreTrainedModel):
            Pre-trained transformer model.
        max_len (int): Maximum length of the input sequence.

    Returns:
        model (transformers.modeling_tf_utils.TFPreTrainedModel):
            Binary classifier model.
    """
    # Define weights initializer
    weights_initializer = tf.keras.initializers.GlorotNormal(seed=42)

    # Define input layers
    input_ids_layer = tf.keras.layers.Input(
        shape=(max_len,), dtype=tf.int32, name="input_ids"
    )
    input_attention_layer = tf.keras.layers.Input(
        shape=(max_len,), dtype=tf.int32, name="attention_mask"
    )

    last_hidden_state = transformer([input_ids_layer, input_attention_layer])[0]

    # Get the last hidden state of the [CLS] token
    cls_token = last_hidden_state[:, 0, :]

    # Define output layer with sigmoid activation function for binary classification
    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        kernel_initializer=weights_initializer,
        kernel_constraint=None,
        bias_initializer="zeros",
    )(cls_token)
    model = tf.keras.models.Model(
        inputs=[input_ids_layer, input_attention_layer], outputs=output
    )
    return model
