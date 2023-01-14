"""
Module for handling parameters
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainParams:
    train_path: str = field(metadata={"help": "Path to the train set."})
    valid_path: str = field(metadata={"help": "Path to the valid set."})
    output_dir: str = field(
        metadata={"help": "Output directory for the training artifacts."}
    )
    target_label: str = field(metadata={"help": "Target label to use."})
    text_column: str = field(metadata={"help": "Column containing the text to use."})
    remove_columns: Optional[List[str]] = field(
        default_factory=list, metadata={"help": "Columns to remove from the dataset."}
    )
    augmented_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the augmented set."}
    )


@dataclass
class ModelParams:
    model_name: str = field(metadata={"help": "Name of the model to use."})
    model_output_dir: str = field(metadata={"help": "Output directory for the model."})
    model_save_name: Optional[str] = field(
        default=None, metadata={"help": "Name of the model to save."}
    )
    hub_model_id: Optional[str] = field(
        default=None, metadata={"help:": "Your huggingface-hub username. Default: None"}
    )
    add_layers: Optional[bool] = field(
        default=False, metadata={"help": "If add input and binary output layers."}
    )
    num_labels: Optional[int] = field(
        default=2, metadata={"help": "Number of labels to use. Default: 2"}
    )

    def __post_init__(self):
        if self.model_save_name is None:
            self.model_save_name = self.model_name


@dataclass
class Hyperparameters:
    epochs: int = field(metadata={"help": "Number of training epochs."})
    learning_rate: float = field(metadata={"help": "Learning rate to use."})
    dropout: float = field(metadata={"help": "Dropout to use."})
    attention_dropout: float = field(metadata={"help": "Attention dropout to use."})
    metric: str = field(metadata={"help": "Metric to use for the best model."})
    max_length: int = field(metadata={"help": "Maximum length of the input sequence."})
    padding: str = field(metadata={"help": "Padding to use."})
    batch_size: int = field(metadata={"help": "Batch size for training."})
    weight_decay: float = field(metadata={"help": "Weight decay to use."})
    warmup_steps: int = field(metadata={"help": "Number of warmup steps."})
    save_strategy: str = field(
        metadata={"help": "Strategy to use for saving the model."}
    )
    evaluation_strategy: str = field(metadata={"help": "Evaluation strategy to use."})
    eval_steps: int = field(metadata={"help": "Number of steps to evaluate the model."})
    load_best_model_at_end: bool = field(
        metadata={"help": "Load the best model at the end."}
    )
    greater_is_better: bool = field(
        metadata={"help": "Whether the metric is greater is better."}
    )


@dataclass
class Params:
    train_params: TrainParams
    model_params: ModelParams
    hyperparameters: Hyperparameters


def get_params(json_file_path) -> Params:
    """
    Load parameters from json file
    Args:
        json_file_path (str): Path to the json file

    Returns:
        params (Params): Parameters
    """
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as f:
            json_loader = json.load(f)
        train_params = TrainParams(**json_loader["train_params"])
        model_params = ModelParams(**json_loader["model_params"])
        hyperparameters = Hyperparameters(**json_loader["hyperparameters"])
        params = Params(
            train_params=train_params,
            model_params=model_params,
            hyperparameters=hyperparameters,
        )
    else:
        raise FileNotFoundError(f"File {json_file_path} not found.")
    return params
