"""
Model selector class for selecting different model types based on input data,
training and such. You will only need to modify the model_selector function
within this class.
"""
#pylint: disable=import-outside-topleve

def get_model_and_tokenizer(self) -> object:
    """Selects the model type based on the model_name and num_label.
    Returns:
        model (transformers.modeling_tf_utils.TFPreTrainedModel): Model to use for training.
        tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizerBase):
            Tokenizer to use for encoding the data.
    """
    if "distilbert" in self.model_name:
        from transformers import (
            TFDistilBertModel,
            DistilBertTokenizer,
            DistilBertConfig,
        )
        config = DistilBertConfig(num_labels=self.num_label)
        model = TFDistilBertModel.from_pretrained(self.model_name, config=config)
        tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

    elif "TFAutoModel" in self.model_name:
        from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
        model = TFAutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_label
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    else:
        raise ValueError("Model not supported.")
    return model, tokenizer
