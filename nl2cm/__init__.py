"""
NL2CM Package

This package implements the NL2CM translation model based on the vec2vec approach.
It provides tools for translating between Natural Language and Conceptual Model
embedding spaces using adversarial training and geometry preservation.
"""

from .model import NL2CMTranslator, Adapter, SharedBackbone, Discriminator
from .training import NL2CMTrainer
from .evaluation import NL2CMEvaluator
from .data_loader import NL2CMDataset, PairedNL2CMDataset, load_nl2cm_data, create_evaluation_splits
from .tensorboard_logger import NL2CMTensorBoardLogger, create_tensorboard_logger

__version__ = "1.0.0"
__author__ = "NL2CM Team"

__all__ = [
    "NL2CMTranslator",
    "Adapter",
    "SharedBackbone",
    "Discriminator",
    "NL2CMTrainer",
    "NL2CMEvaluator",
    "NL2CMDataset",
    "PairedNL2CMDataset",
    "load_nl2cm_data",
    "create_evaluation_splits",
    "NL2CMTensorBoardLogger",
    "create_tensorboard_logger"
]
