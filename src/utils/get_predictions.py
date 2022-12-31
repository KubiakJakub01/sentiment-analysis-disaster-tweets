"""
Make predictions on test data and save to final csv file using a trained model.
"""
import argparse

from transformers import pipeline


def get_params():
    """Get parameters from command line.

    Returns:
        args (argparse.Namespace): Arguments from command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        help="Path to checkpoint with the model to use.",
    )
    parser.add_argument(
        "--path_to_test_data",
        "-t",
        type=str,
        default="data/test.csv",
        help="Path to the test set.",
    )
    parser.add_argument(
        "--num_labels",
        "-n",
        type=int,
        default=2,
        help="Number of labels for the model.",
    )
    parser.add_argument(
        "--save_predictions_path", 
        "-s", type=str, 
        help="Path to save the predictions."
    )
    return parser.parse_args()

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