"""
Main training script for the project.

Usage:
    python3 -m src.train src/config/params.json
"""
# pylint: disable=too-many-arguments
# pylint: disable=redefined-outer-name
# pylint: disable=unused-variable

# Import basic libraries
import os
import sys
from datetime import datetime
from pathlib import Path

# Import huggingface libraries
from datasets import load_dataset
from keras.callbacks import TensorBoard

# Load the DistilBERT tokenizer to process the text field:
from transformers import DataCollatorWithPadding, create_optimizer

# Import metrics
from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback

# Import model and tokenizer selector class
from src.model.nlp_models_selector import get_model_and_tokenizer

# Import metrics
from src.utils.nlp_metric import Metric

# Import modules from src
from src.utils.params import get_params

# Import utils for text cleaning
from src.utils.text_cleaning import text_cleaning


def load_dataset_from_csv(train_path: str, valid_path: str, augument_path: str) -> None:
    """Load the train and valid sets.

    Args:
        train_path (str): Path to the train set.
        valid_path (str): Path to the valid set.
        augument_path (str): Path to the augumented set.

    Returns:
        dataset (dict): Dictionary containing the train and valid sets."""
    if augument_path:
        # load only text and label columns
        dataset = load_dataset(
            "csv",
            data_files={"train": [train_path, augument_path], "validation": valid_path},
        )
    else:
        dataset = load_dataset(
            "csv", data_files={"train": train_path, "validation": valid_path}
        )
    return dataset


def preprocess_data(dataset: dict, text_column: str = "text") -> dict:
    """Preprocess the data.

    Args:
        dataset (dict): Dictionary containing the train and valid sets.
        text_column (str): Name of the text column.

    Returns:
        dataset (dict): Dictionary containing cleaned the train and valid sets."""
    dataset = dataset.map(
        lambda examples: {text_column: [text_cleaning(examples[text_column])]}
    )
    return dataset


def tokenize_text(text: str):
    """Tokenize the data.

    Args:
        text (str): Text to tokenize.

    Returns:
        tokenized_data (dict): Dictionary containing the tokenized data."""
    return tokenizer(
        text["text"],
        truncation=True,
        is_split_into_words=True,
        padding="longest",
        return_attention_mask=True,
        return_token_type_ids=False,
    )


def prepare_dataset(dataset, columns, label_cols, batch_size, shuffle, collate_fn):
    """Prepare the dataset for training.

    Args:
        dataset (dict): Dictionary containing the train and valid sets.
        columns (list): List of columns to use.
        label_cols (list): List of labels to use.
        batch_size (int): Batch size to use.
        shuffle (bool): Whether to shuffle the data.
        collate_fn (function): Function to use for collating the data.

    Returns:
        dataset (dict): Dictionary containing the train and valid sets."""
    tf_dataset = dataset.to_tf_dataset(
        columns=columns,
        label_cols=label_cols,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    return tf_dataset


def prepare_callbacks(
    hiperparameters,
    tokenizer,
    metric,
    valid_dataset,
    target_label,
    model_output_dir,
    model_save_name,
    hub_model_id,
    log_dir,
):
    """Prepare the callbacks for training.

    Args:
        hiperparameters (dict): Dictionary containing the hiperparameters.
        tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizerBase):
            Tokenizer to use for encoding the data.
        metric (transformers.metric.Metric): Metric to use for training.
        valid_dataset (tf.data.Dataset): Dataset to use for validation.
        target_label (list): List of labels to use.
        model_output_dir (str): Path to the output directory.
        model_save_name (str): Name of the model to save.
        hub_model_id (str): Model id to use for saving the model to the hub.
        log_dir (str): Path to the log directory.

    Returns:
        callbacks (list): List containing the callbacks for training."""
    keras_metric_callback = KerasMetricCallback(
        metric_fn=metric.compute_metrics,
        eval_dataset=valid_dataset,
        batch_size=hiperparameters.batch_size,
        label_cols=target_label,
    )
    tensorboard_callback = TensorBoard(log_dir=log_dir)

    if hub_model_id != None:
        push_to_hub_callback = PushToHubCallback(
            output_dir=model_output_dir,
            tokenizer=tokenizer,
            save_strategy=hiperparameters.save_strategy,
            hub_model_id=f"{hub_model_id}/{model_save_name}",
        )
        callbacks = [keras_metric_callback, push_to_hub_callback, tensorboard_callback]
    else:
        callbacks = [keras_metric_callback, tensorboard_callback]
    return callbacks


def train():
    """Pipeline for training the model.

    Args:
        params (dict): Dictionary of parameters.

    Returns:
        Saves the trained model to the output directory.
    """

    # Load the train and valid sets
    dataset = load_dataset_from_csv(
        train_path=params.train_params.train_path,
        valid_path=params.train_params.valid_path,
        augument_path=params.train_params.augmented_path,
    )

    # Preprocess the data
    dataset = preprocess_data(dataset)

    # Tokenize the data
    tokenized_dataset = dataset.map(tokenize_text, batched=True)

    # Load data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

    # Prepare the dataset for training
    tf_train_dataset = prepare_dataset(
        tokenized_dataset["train"],
        columns=["input_ids", "attention_mask", params.train_params.target_label],
        label_cols=[params.train_params.target_label],
        batch_size=params.hyperparameters.batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )

    # Prepare the dataset for validation
    tf_valid_dataset = prepare_dataset(
        tokenized_dataset["validation"],
        columns=["input_ids", "attention_mask", params.train_params.target_label],
        label_cols=[params.train_params.target_label],
        batch_size=params.hyperparameters.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    # Set up the optimizer
    optimizer, schedule = create_optimizer(
        init_lr=params.hyperparameters.learning_rate,
        num_warmup_steps=params.hyperparameters.warmup_steps,
        num_train_steps=int(
            (len(tf_train_dataset) // params.hyperparameters.batch_size)
            * params.hyperparameters.epochs
        ),
    )

    # Set up the metric
    metric = Metric(params.hyperparameters.metric)

    # Compile the model
    model.compile(optimizer=optimizer)

    # Set up the callbacks
    callbacks = prepare_callbacks(
        hiperparameters=params.hyperparameters,
        tokenizer=tokenizer,
        metric=metric,
        valid_dataset=tf_valid_dataset,
        target_label=[params.train_params.target_label],
        model_output_dir=params.model_params.model_output_dir,
        model_save_name=params.model_params.model_save_name,
        hub_model_id=params.model_params.hub_model_id,
        log_dir=params.train_params.output_dir,
    )

    # Fit the model
    model.fit(
        tf_train_dataset,
        validation_data=tf_valid_dataset,
        epochs=params.hyperparameters.epochs,
        callbacks=callbacks,
        use_multiprocessing=True,
    )


if __name__ == "__main__":

    # Get start time
    start_time = datetime.now()
    start_time = start_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Load parameters from config json file
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Load the parameters from the config file
        params = get_params(sys.argv[1])
    else:
        print(
            """No config file provided. Specify a config file. 
            Check example config file in the config folder: src/config/params.json.
            Or look at the README for more information."""
        )
        sys.exit(1)

    # Load model, and tokenizer
    model, tokenizer = get_model_and_tokenizer(
        model_name=params.model_params.model_name,
        num_labels=params.model_params.num_labels,
        add_layers=params.model_params.add_layers,
        droput=params.hyperparameters.dropout,
        att_droput=params.hyperparameters.attention_dropout,
        max_length=params.hyperparameters.max_length,
    )

    # Create the output directory
    params.train_params.output_dir = (
        Path(params.train_params.output_dir)
        / f"{params.model_params.model_save_name}_{start_time}"
    )
    os.makedirs(params.train_params.output_dir, exist_ok=True)

    # Create model output directory
    os.makedirs(params.model_params.model_output_dir, exist_ok=True)

    # Train the model
    train()
