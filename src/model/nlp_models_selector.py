"""
Model selector class for selecting different model types based on input data,
training and such. You will only need to modify the model_selector function
within this class.
"""
#pylint: disable=import-outside-topleve

def get_model_and_tokenizer(model_name: str, num_label: int) -> object:
    """Selects the model type based on the model_name and num_label.

    Args:
        model_name (str): Name of the model to use.
        num_label (int): Number of labels to use.
        
    Returns:
        model (transformers.modeling_tf_utils.TFPreTrainedModel): Model to use for training.
        tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizerBase):
            Tokenizer to use for encoding the data.
    """
    if "distilbert" in model_name:
        from transformers import (
            TFDistilBertModel,
            DistilBertTokenizer,
            DistilBertConfig,
        )
        config = DistilBertConfig(num_labels=num_label)
        model = TFDistilBertModel.from_pretrained(model_name, config=config)
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    elif "TFAutoModel" in model_name:
        from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
        model = TFAutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_label
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    else:
        raise ValueError("Model not supported.")
    return model, tokenizer
