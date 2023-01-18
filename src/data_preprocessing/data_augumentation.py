"""
Augument data by adding noise to the data

Usage:
    python src/data_preprocessing/data_augumentation.py [-p data/train.csv] \
                                                    [-s data/augumented_train.csv] \
                                                    [-m models/bert-base-uncased] \
                                                    [-a substitute] \
                                                    [-n 2] \
                                                    [--aug_min 1] \
                                                    [--aug_p 0.3]
"""

# Import basic libraries
import argparse

import nlpaug.augmenter.word as nlpaw
import pandas as pd
import tqdm


def get_params():
    """Get parameters from command line.

    Returns:
        args (argparse.Namespace): Arguments from command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_data",
        "-p",
        type=str,
        default="data/train.csv",
        help="Path to the data to augument.",
    )
    parser.add_argument(
        "--path_to_save_data",
        "-s",
        type=str,
        default="data/augumented_train.csv",
        help="Path to save the augumented data.",
    )
    parser.add_argument(
        "--model_name", "-m", type=str, default="bert-base-uncased", help="Model name."
    )
    parser.add_argument(
        "--augumentation_type",
        "-a",
        type=str,
        default="substitute",
        help="Augumentation type to use.",
    )
    parser.add_argument(
        "--num_threads", "-n", type=int, default=4, help="Number of threads to use."
    )
    parser.add_argument(
        "--aug_min", type=int, default=1, help="Minimum number of words to augment."
    )
    parser.add_argument(
        "--aug_p", type=float, default=0.3, help="Probability of words to augment."
    )
    return parser.parse_args()


def augment_sentence(sentence, aug, num_threads):
    """Augment a sentence.

    Args:
        sentence (str): Sentence to augment.
        aug (nlpaug.augmenter.word.WordAugmenter): Augmenter to use.
        num_threads (int): Number of threads to use.

    Returns:
        augmented_sentence (str): Augmented sentence."""
    augmented_sentence = aug.augment(sentence, n=num_threads)
    return augmented_sentence


def augument_text(df, aug, num_threads):
    """Augment text.

    Args:
        df (pd.DataFrame): Dataframe containing the text to augment.
        aug (nlpaug.augmenter.word.WordAugmenter): Augmenter to use.
        num_threads (int): Number of threads to use.

    Returns:
        df (pd.DataFrame): Dataframe containing the augmented text."""
    # Augument text with progress bar
    df["augument_text"] = tqdm.tqdm(
        df["text"].apply(lambda x: augment_sentence(x, aug, num_threads))
    )
    return df


def augument_data(df, aug, num_threads):
    """Augment data.

    Args:
        df (pd.DataFrame): Dataframe containing the text to augment.
        aug (nlpaug.augmenter.word.WordAugmenter): Augmenter to use.
        num_threads (int): Number of threads to use.

    Returns:
        df (pd.DataFrame): Dataframe containing the augmented text."""
    df = augument_text(df, aug, num_threads)
    df = df[["text", "labels", "augument_text"]]
    df = df.rename(columns={"text": "original_text"})
    df = df.rename(columns={"augument_text": "text"})
    return df


if __name__ == "__main__":

    # Get the parameters
    args = get_params()

    print(f"Loading data from: {args.path_to_data}")
    # Read the data
    df = pd.read_csv(args.path_to_data)

    # Augument the data
    print(
        f"Augumenting data using: {args.augumentation_type} and model: {args.model_name}"
    )
    aug = nlpaw.ContextualWordEmbsAug(
        model_path=args.model_name,
        action=args.augumentation_type,
        aug_min=args.aug_min,
        aug_p=args.aug_p,
    )

    print("Augumenting data...")
    df = augument_data(df=df, aug=aug, num_threads=args.num_threads)

    # Save the data
    print(f"Saving data to: {args.path_to_save_data}")
    df.to_csv(args.path_to_save_data, index=False)
