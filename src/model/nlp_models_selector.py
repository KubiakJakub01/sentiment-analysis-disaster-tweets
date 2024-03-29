"""
Model selector class for selecting different model types based on input data,
training and such. You will only need to modify the model_selector function
within this class.
"""
# pylint: disable=import-outside-topleve

import logging
from src.model.utils.build_custom_transformer import add_input_and_binary_output_layers

# Get the logger
logger = logging.getLogger(__name__)

def get_model_and_tokenizer(
    model_name: str,
    num_labels: int = 2,
    droput: float = 0.2,
    att_droput: float = 0.2,
    max_length: int = 128,
    add_layers: bool = False,
) -> object:
    """Selects the model type based on the model_name and num_label.

    Args:
        model_name (str): Name of the model to use.
        num_label (int): Number of labels to use.
        droput (float): Dropout to use.
        att_droput (float): Attention dropout to use.

    Returns:
        model (transformers.modeling_tf_utils.TFPreTrainedModel): Model to use for training.
        tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizerBase):
            Tokenizer to use for encoding the data.
    """
    logger.info("Loading model: %s", model_name)
    if add_layers:
        from transformers import (
            DistilBertConfig,
            DistilBertTokenizerFast,
            TFDistilBertModel,
        )

        config = DistilBertConfig(
            num_labels=num_labels, dropout=droput, attention_dropout=att_droput
        )
        model = TFDistilBertModel.from_pretrained(model_name, config=config)
        logger.info("Adding input and binary output layers.")
        model = add_input_and_binary_output_layers(model, max_length)
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    elif "distilbert" in model_name:
        from transformers import (
            DistilBertConfig,
            DistilBertTokenizerFast,
            TFDistilBertForSequenceClassification,
        )

        config = DistilBertConfig(num_labels=num_labels)
        model = TFDistilBertForSequenceClassification.from_pretrained(
            model_name, config=config
        )
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    elif "TFAutoModel" in model_name:
        from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

        model = TFAutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    else:
        logger.error("Model not supported.")
        raise ValueError("Model not supported.")
    return model, tokenizer


def model_fn(features, labels, mode, params):
    import tensorflow as tf
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

    # Create the model and tokenizer
    model_type = params["model_type"]
    model = TFAutoModelForSequenceClassification.from_pretrained(model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_type)

    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])

    # Compile the model
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    # Return the model
    return model
