"""
Main inference script for the project.

Run example:
python src/eval.py -m models/bert-base-uncased \ 
                    -t data/test.csv \
                    -s results \
                    -n 2 \ 
                    -s results \
                    -e accuracy precision recall f1 \
"""

# Imports basic libraries
import argparse
import json
from datetime import datetime
from pathlib import Path

# Import huggingface libraries
from datasets import load_dataset

# Import modules
from src.model.nlp_models_selector import get_model_and_tokenizer
from src.utils.compute_results import get_results, save_results
from src.utils.get_predictions import get_prdiction
from src.utils.nlp_metric import Metric


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
    parser.add_argument(
        "--metrics",
        "-e",
        nargs="+",
        type=str,
        help="Metrics to use.",
        default=["accuracy", "precision", "recall", "f1"],
    )
    return parser.parse_args()



def save_predictions(preds, save_predictions_path):
    """Save predictions to json file.

    Args:
        preds (dict): Dictionary with the predictions.
        save_predictions_path (str): Path to save the predictions.

    Returns:
        Create a file with the predictions."""
    if save_predictions_path:
        save_predictions_path.mkdir(parents=True, exist_ok=True)
        with open(
            save_predictions_path / "predictions.json", "w", encoding="utf-8"
        ) as json_file:
            json.dump(preds, json_file)


def evaluate():
    """Evaluate the model."""

    # Get predictions
    preds = get_prdiction(model, tokenizer, test_dataset["text"])

    # Save predictions
    save_predictions(preds, SAVE_PREDICTIONS_PATH)

    # Compute metrics
    results = get_results(preds, test_dataset["labels"], metrics)

    # Print metric results
    print(results)

    # Save metric results
    save_results(results, SAVE_PREDICTIONS_PATH)


if __name__ == "__main__":

    # Get parameters
    params = get_params()

    # Define constant variables
    MODEL_PATH = Path(params.model_path)
    MODEL_NAME = MODEL_PATH.name
    NUM_LABELS = params.num_labels
    START_TIME = datetime.now()
    START_TIME = START_TIME.strftime("%Y-%m-%d_%H-%M-%S")
    SAVE_PREDICTIONS_PATH = Path(params.save_predictions_path) / MODEL_NAME / START_TIME

    print(f"Save predictions path: {SAVE_PREDICTIONS_PATH}")

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, NUM_LABELS)

    # Load test data
    test_dataset = load_dataset("csv", data_files={"test": params.path_to_test_data})
    test_dataset = test_dataset["test"]

    # Load metric
    print(f"Metrices: {params.metrics}")
    metrics = [Metric(metric) for metric in params.metrics]

    # Evaluate the model
    evaluate()
