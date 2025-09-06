import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import random


class PairEmbeds(Dataset):
    """Loads paired (source, target) vectors from .npy files."""
    def __init__(self, X, Y, norm=True):
        assert X.shape[0] == Y.shape[0], "Mismatched pair counts."
        if norm:
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):  return self.x.shape[0]
    def __getitem__(self, i): return self.x[i], self.y[i]


def split_indices(n, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_ratio))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def l2(x,y): return F.mse_loss(x,y)

def cosine_loss(a,b):
    a = F.normalize(a, dim=-1); b = F.normalize(b, dim=-1)
    return 1 - (a*b).sum(dim=-1).mean()

def vsp_loss(x, Fx):
    x  = F.normalize(x,  dim=-1)
    Fx = F.normalize(Fx, dim=-1)
    Gx  = x @ x.t()
    GFx = Fx @ Fx.t()
    return F.mse_loss(Gx, GFx)

def cosine_sim_matrix(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A @ B.T  # (NA, NB)

def topk_and_mrr(sim, assume_paired=True):
    """
    sim: (N, N) similarity matrix (rows: queries; cols: gallery).
    If paired, the correct index for row i is i.
    Returns Top1, Top5, MRR, mean rank.
    """
    N = sim.shape[0]
    ranks = []
    top1 = 0
    top5 = 0
    # Argsort descending per row
    order = np.argsort(-sim, axis=1)
    for i in range(N):
        correct = i if assume_paired else None  # extend here if you track indices another way
        row = order[i]
        # find position of correct
        pos = int(np.where(row == correct)[0][0])
        ranks.append(pos + 1)  # 1-based
        if pos == 0: top1 += 1
        if pos < 5: top5 += 1
    ranks = np.array(ranks)
    mrr = np.mean(1.0 / ranks)
    return {
        "top1": top1 / N,
        "top5": top5 / N,
        "mrr": float(mrr),
        "mean_rank": float(np.mean(ranks))
    }

def geometry_corr(A, B):
    """
    Pearson correlation between upper triangles of pairwise cosine sims of A and B.
    Measures how well neighborhoods are preserved after translation.
    """
    SA = cosine_sim_matrix(A, A)
    SB = cosine_sim_matrix(B, B)
    iu = np.triu_indices(SA.shape[0], k=1)
    a = SA[iu]; b = SB[iu]
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float((a*b).mean())
