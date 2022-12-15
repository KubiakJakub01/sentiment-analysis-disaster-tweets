"""
Main training script for the project.
"""

# Import basic libraries
import argparse
from datetime import datetime
from datasets import load_dataset

# Load the DistilBERT tokenizer to process the text field:
from transformers import (
    DataCollatorWithPadding,
    create_optimizer,
)

# Import metrics
from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback
from keras.callbacks import TensorBoard


