"""
CMVec2Vec: A library for translating between NL and CM embedding spaces using the vec2vec approach.

This library provides a simplified interface for Natural Language to Conceptual Model
embedding translation, removing the multi-domain complexity and focusing on the core
NL2CM translation task.
"""

from .translators.cm_vec2vec_translator import CMVec2VecTranslator
from .training import CMVec2VecTrainer
from .evaluation import CMVec2VecEvaluator
from .data_loader import load_nl2cm_data
from .config import load_config, save_config
from .losses_cosine_distance import (
    adversarial_loss,
    reconstruction_loss,
    cycle_consistency_loss,
    vector_space_preservation_loss,
    compute_all_losses
)

__version__ = "1.0.0"
__author__ = "CMVec2Vec Team"

__all__ = [
    "CMVec2VecTranslator",
    "CMVec2VecTrainer",
    "CMVec2VecEvaluator",
    "load_nl2cm_data",
    "load_config",
    "save_config",
    "adversarial_loss",
    "reconstruction_loss",
    "cycle_consistency_loss",
    "vector_space_preservation_loss",
    "compute_all_losses",
]
