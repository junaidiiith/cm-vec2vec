"""
NL2CM Data Loading and Preprocessing Module

This module handles loading and preprocessing of Natural Language (NL) and Conceptual Model (CM) 
embeddings for the NL2CM translation task, following the vec2vec approach.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict
import random
from sklearn.model_selection import train_test_split

from nl2cm.embed import get_embeddings


class NL2CMDataset(Dataset):
    """
    Dataset for NL2CM translation task.

    This dataset provides unpaired NL and CM embeddings for training the vec2vec translator.
    The embeddings are from the same encoder but represent different domains (natural language
    vs conceptual modeling language).
    """

    def __init__(self, nlt_embeddings: np.ndarray, cmt_embeddings: np.ndarray,
                 normalize: bool = True, noise_level: float = 0.0):
        """
        Initialize the dataset.

        Args:
            nlt_embeddings: Array of NL embeddings (N, d)
            cmt_embeddings: Array of CM embeddings (M, d) 
            normalize: Whether to normalize embeddings
            noise_level: Level of noise to add during training
        """
        self.nlt_embeddings = nlt_embeddings
        self.cmt_embeddings = cmt_embeddings
        self.normalize = normalize
        self.noise_level = noise_level
        self.training = True  # Default to training mode

        if normalize:
            self.nlt_embeddings = self._normalize_embeddings(
                self.nlt_embeddings)
            self.cmt_embeddings = self._normalize_embeddings(
                self.cmt_embeddings)

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)

    def __len__(self) -> int:
        return max(len(self.nlt_embeddings), len(self.cmt_embeddings))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        # Randomly sample from each domain to ensure unpaired data
        nlt_idx = random.randint(0, len(self.nlt_embeddings) - 1)
        cmt_idx = random.randint(0, len(self.cmt_embeddings) - 1)

        nlt_emb = self.nlt_embeddings[nlt_idx]
        cmt_emb = self.cmt_embeddings[cmt_idx]

        # Add noise during training
        if self.training and self.noise_level > 0:
            nlt_emb = nlt_emb + \
                np.random.normal(0, self.noise_level, nlt_emb.shape)
            cmt_emb = cmt_emb + \
                np.random.normal(0, self.noise_level, cmt_emb.shape)

            # Renormalize after adding noise
            if self.normalize:
                nlt_emb = nlt_emb / (np.linalg.norm(nlt_emb) + 1e-8)
                cmt_emb = cmt_emb / (np.linalg.norm(cmt_emb) + 1e-8)

        return {
            'nlt': torch.FloatTensor(nlt_emb),
            'cmt': torch.FloatTensor(cmt_emb)
        }


class PairedNL2CMDataset(Dataset):
    """
    Dataset for paired NL-CM data (used for evaluation only).

    This dataset provides paired NL and CM embeddings for evaluation purposes.
    """

    def __init__(self, nlt_embeddings: np.ndarray, 
        cmt_embeddings: np.ndarray, normalize: bool = True,
        shuffle: bool = True
    ):
        """
        Initialize the paired dataset.

        Args:
            nlt_embeddings: Array of NL embeddings (N, d)
            cmt_embeddings: Array of CM embeddings (N, d) - same length as NL
            normalize: Whether to normalize embeddings
        """
        assert len(nlt_embeddings) == len(
            cmt_embeddings), "Paired data must have same length"

        self.nlt_embeddings = nlt_embeddings
        self.cmt_embeddings = cmt_embeddings
        self.normalize = normalize

        if normalize:
            self.nlt_embeddings = self._normalize_embeddings(
                self.nlt_embeddings)
            self.cmt_embeddings = self._normalize_embeddings(
                self.cmt_embeddings)

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)

    def __len__(self) -> int:
        return len(self.nlt_embeddings)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a paired sample from the dataset."""
        return {
            'nlt': torch.FloatTensor(self.nlt_embeddings[idx]),
            'cmt': torch.FloatTensor(self.cmt_embeddings[idx])
        }


def load_nl2cm_data(data_path: str, nl_cm_cols: list[str], test_size: float = 0.2,
                    random_state: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load NL2CM data and create train/validation/test splits.

    Args:
        data_path: Path to the pickle file containing the dataframe
        test_size: Fraction of data to use for testing
        random_state: Random state for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load the dataframe
    nlt_embeddings, cmt_embeddings = get_embeddings(data_path, nl_cm_cols)

    print(
        f"Loaded {len(nlt_embeddings)} NL embeddings and {len(cmt_embeddings)} CM embeddings")
    print(f"Embedding dimension: {nlt_embeddings.shape[1]}")

    # Create unpaired datasets for training
    train_nlt, test_nlt = train_test_split(nlt_embeddings, test_size=test_size,
                                           random_state=random_state)
    train_cmt, test_cmt = train_test_split(cmt_embeddings, test_size=test_size,
                                           random_state=random_state)

    # Further split training data for validation
    train_nlt, val_nlt = train_test_split(train_nlt, test_size=0.2,
                                          random_state=random_state)
    train_cmt, val_cmt = train_test_split(train_cmt, test_size=0.2,
                                          random_state=random_state)

    # Create datasets
    train_dataset = NL2CMDataset(train_nlt, train_cmt, normalize=True)
    val_dataset = NL2CMDataset(val_nlt, val_cmt, normalize=True)
    test_dataset = PairedNL2CMDataset(test_nlt, test_cmt, normalize=True)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32,
                            shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32,
                             shuffle=False, num_workers=4)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'embedding_dim': nlt_embeddings.shape[1]
    }


def create_evaluation_splits(data_path: str, nl_cm_cols: list[str], n_eval_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create evaluation splits for computing vec2vec-style metrics.

    Args:
        data_path: Path to the pickle file containing the dataframe
        n_eval_samples: Number of samples to use for evaluation

    Returns:
        Tuple of (nlt_eval, cmt_eval) arrays
    """
    nlt_embeddings, cmt_embeddings = get_embeddings(data_path, nl_cm_cols)

    # Sample for evaluation
    n_samples = min(n_eval_samples, len(nlt_embeddings))
    indices = np.random.choice(len(nlt_embeddings), n_samples, replace=False)

    nlt_eval = nlt_embeddings[indices]
    cmt_eval = cmt_embeddings[indices]

    return nlt_eval, cmt_eval