"""
Compute results based on the predictions and labels.
"""
# Imports basic libraries
import argparse
import json
from pathlib import Path

import pandas as pd

# Import modules
from src.utils.nlp_metric import Metric
