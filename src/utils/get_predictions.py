"""
Make predictions on test data and save to final csv file using a trained model.
"""
from transformers import pipeline


def map_label_to_integers(label):
    """Map labels to integers."""
    label = 1 if "1" in label else 0
    return label
    

def get_prdiction(model, tokenizer, text):
    """Get predictions for the model.

    Args:
        model (transformers.modeling_tf_utils.TFPreTrainedModel): Model to use for training.
        tokenizer (transformers.PreTrainedTokenizerBase): Tokenizer to use for encoding the data.
        text (list): List of text to predict.

    Returns:
        preds (dict): Dictionary with the predictions."""
    clasificator = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)
    preds = clasificator(text)

    preds = [
        {
            "score": round(pred["score"], 4),
            "labels": map_label_to_integers(pred["label"]),
        }
        for pred in preds
    ]

    return preds