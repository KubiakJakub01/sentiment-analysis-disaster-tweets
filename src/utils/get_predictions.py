"""
Make predictions on test data and save to final csv file using a trained model.

Usage:
    python3 -m src.utils.get_predictions -m <model_path> -t <path_to_test_data> -n <num_labels> -s <save_predictions_path>
"""
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import pipeline

from src.model.nlp_models_selector import get_model_and_tokenizer


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
        "--save_predictions_path", "-s", type=str, help="Path to save the predictions."
    )
    return parser.parse_args()


def map_label_to_integers(label):
    """Map labels to integers."""
    label = 1 if "1" in label else 0
    return label


def get_prdiction(model, tokenizer, id_list, text_list):
    """Get predictions for the model.

    Args:
        model (transformers.modeling_tf_utils.TFPreTrainedModel): Model to use for training.
        tokenizer (transformers.PreTrainedTokenizerBase): Tokenizer to use for encoding the data.
        text (list): List of text to predict.

    Returns:
        preds (dict): Dictionary with the predictions."""
    preds = []
    clasificator = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)
    for pred in tqdm(clasificator(text_list), total=len(text_list), desc="Predictions"):
        preds.append(
            {
                "id": id_list,
                "score": pred["score"],
                "labels": map_label_to_integers(pred["label"]),
            }
        )

    return preds


def save_predictions(preds, save_predictions_path):
    """Save predictions to json file.

    Args:
        preds (dict): Dictionary with the predictions.
        save_predictions_path (str): Path to save the predictions.

    Returns:
        Create a file with the predictions."""
    # Save id and labels to csv file
    df = pd.DataFrame(preds[["id", "labels"]])
    df.to_csv(save_predictions_path, index=False)


if __name__ == "__main__":
    # Get parameters from command line
    args = get_params()

    # Define constant variables
    MODEL_PATH = Path(args.model_path)
    MODEL_NAME = MODEL_PATH.name
    NUM_LABELS = args.num_labels
    START_TIME = datetime.now()
    START_TIME = START_TIME.strftime("%Y-%m-%d_%H-%M-%S")
    SAVE_PREDICTIONS_PATH = Path(args.save_predictions_path) / MODEL_NAME / START_TIME

    # Load test data
    df = pd.read_csv(args.path_to_test_data)

    # Get model and tokenizer
    print("Loading model...")
    model, tokenizer = get_model_and_tokenizer(
        model_name=str(MODEL_PATH), num_labels=NUM_LABELS
    )

    # Get predictions
    print("Making predictions...")
    preds = get_prdiction(model, tokenizer, df["id"].to_list(), df["text"].tolist())

    # Save predictions
    print("Saving predictions...")
    Path(SAVE_PREDICTIONS_PATH).parent.mkdir(parents=True, exist_ok=True)
    save_predictions(preds, SAVE_PREDICTIONS_PATH)

    print("Done!")
