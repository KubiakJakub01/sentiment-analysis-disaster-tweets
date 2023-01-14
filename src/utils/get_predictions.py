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
from src.utils.text_cleaning import text_cleaning


def get_params():
    """Get parameters from command line.

    Returns:
        args (argparse.Namespace): Arguments from command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", "-m", type=str, help="Path to checkpoint with the model to use."
    )
    parser.add_argument(
        "--save_predictions_path", "-s", type=str, help="Path to save the predictions."
    )
    parser.add_argument(
        "--path_to_test_data",
        "-t",
        type=str,
        default="data/test.csv",
        help="Path to the test set.",
    )
    parser.add_argument("--batch_size", "-b", type=int, default=8)
    parser.add_argument(
        "--num_labels",
        "-n",
        type=int,
        default=2,
        help="Number of labels for the model.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of the text column in the dataset.",
    )
    parser.add_argument(
        "--id_column",
        type=str,
        default="id",
        help="Name of the id column in the dataset.",
    )
    return parser.parse_args()


def map_label_to_integers(label):
    """Map labels to integers."""
    label = 1 if "1" in label else 0
    return label


def get_prdiction(model, tokenizer, task, id_list, text_list, batch_size):
    """Get predictions for the model.

    Args:
        model (transformers.modeling_tf_utils.TFPreTrainedModel): Model to use for training.
        tokenizer (transformers.PreTrainedTokenizerBase): Tokenizer to use for encoding the data.
        task (str): Task to use for the model.
        id_list (list): List with the ids of the samples.
        text_list (list): List with the text of the samples.
        batch_size (int): Batch size to use for the predictions.

    Returns:
        preds (dict): Dictionary with the predictions."""
    preds = []
    clasificator = pipeline(task=task, model=model, tokenizer=tokenizer)
    for i, pred in enumerate(
        tqdm(
            clasificator(text_list, batch_size=batch_size),
            total=len(text_list),
            desc="Predictions",
        )
    ):
        preds.append(
            {
                "id": id_list[i],
                "score": round(pred["score"], 4),
                "target": map_label_to_integers(pred["label"]),
            }
        )

    return preds


def save_predictions(preds, save_predictions_path, id_column, label_column):
    """Save predictions to json file.

    Args:
        preds (dict): Dictionary with the predictions.
        save_predictions_path (str): Path to save the predictions.

    Returns:
        Create a file with the predictions."""
    # Save id and labels to csv file
    # create dataframe from dictionary
    df = pd.DataFrame(preds, columns=[id_column, label_column])
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
    SAVE_PREDICTIONS_PATH = Path(args.save_predictions_path) / MODEL_NAME / f"results_{START_TIME}.csv"

    # Load test data
    df = pd.read_csv(args.path_to_test_data)

    # Get model and tokenizer
    print("Loading model...")
    model, tokenizer = get_model_and_tokenizer(
        model_name=str(MODEL_PATH), num_labels=NUM_LABELS
    )

    # Cleaning data
    print("Cleaning data...")
    df[args.text_column] = df[args.text_column].apply(lambda x: text_cleaning(x))

    # Get predictions
    print("Making predictions...")
    preds = get_prdiction(model=model, 
                        tokenizer=tokenizer, 
                        id_list=df[args.id_column].to_list(), 
                        text_list=df[args.text_column].tolist(),
                        batch_size=args.batch_size)

    # Save predictions
    print("Saving predictions...")
    Path(SAVE_PREDICTIONS_PATH).parent.mkdir(parents=True, exist_ok=True)
    save_predictions(preds=preds, 
                    save_predictions_path=SAVE_PREDICTIONS_PATH,
                    id_column=args.id_column,
                    label_column="target")

    print("Done!")
