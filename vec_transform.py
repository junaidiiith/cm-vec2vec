# vecmap_transformer.py
import os
import math
import json
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from utils import (
    PairEmbeds, 
    cosine_loss, 
    split_indices, 
    cosine_sim_matrix, 
)


# ---------------------------
# Transformer Blocks
# ---------------------------
class TransformerBlock(nn.Module):
    """Self-attention block: LN -> MHA -> residual -> LN -> FFN -> residual."""
    def __init__(self, dim, heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads,
                                          dropout=dropout, batch_first=False)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):  # x: (T,B,D)
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + self.drop1(a)
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x

class XTransformerBlock(nn.Module):
    """Decoder-style block with *cross*-attention: self-attn on q, then cross-attn q<-k/v."""
    def __init__(self, dim, heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.self_ln = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=False)
        self.self_drop = nn.Dropout(dropout)

        self.cross_q_ln = nn.LayerNorm(dim)
        self.cross_kv_ln = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=False)
        self.cross_drop = nn.Dropout(dropout)

        self.ffn_ln = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, q_tokens, kv_tokens):  # both: (T,B,D)
        # self-attn on q_tokens
        qs = self.self_ln(q_tokens)
        s, _ = self.self_attn(qs, qs, qs, need_weights=False)
        x = q_tokens + self.self_drop(s)
        # cross-attn: Q from x, K/V from kv
        qc = self.cross_q_ln(x)
        kvc = self.cross_kv_ln(kv_tokens)
        c, _ = self.cross_attn(qc, kvc, kvc, need_weights=False)
        x = x + self.cross_drop(c)
        # FFN
        x = x + self.ffn_drop(self.ffn(self.ffn_ln(x)))
        return x

# ---------------------------
# Tokenizers
# ---------------------------
class VectorTokenizer(nn.Module):
    """(B, d_in) → (T, B, d_lat) via one linear projection; learned pos embeddings added."""
    def __init__(self, d_in, d_lat, T):
        super().__init__()
        self.T = T
        self.proj = nn.Linear(d_in, T * d_lat)
        self.pos = nn.Parameter(torch.randn(T, 1, d_lat) / math.sqrt(d_lat))
        self.ln = nn.LayerNorm(d_lat)

    def forward(self, v):  # v: (B, d_in)
        B = v.size(0)
        t = self.proj(v).view(B, self.T, -1)          # (B, T, d_lat)
        t = self.ln(t).transpose(0, 1)                # (T, B, d_lat)
        return t + self.pos                           # add learned positions

class VectorDetokenizer(nn.Module):
    """(T, B, d_lat) → (B, d_out). Uses first token (CLS-like) then projects."""
    def __init__(self, d_lat, d_out, use_first_token=True):
        super().__init__()
        self.use_first = use_first_token
        self.proj = nn.Linear(d_lat, d_out)
        self.out_ln = nn.LayerNorm(d_out)

    def forward(self, tokens):  # (T, B, d_lat)
        pooled = tokens[0] if self.use_first else tokens.mean(dim=0)  # (B, d_lat)
        return self.out_ln(self.proj(pooled))                         # (B, d_out)

# ---------------------------
# The Mapper
# ---------------------------
class VecMapTransformer(nn.Module):
    """
    Maps source vector -> target vector with a transformer:
      - Encode source vector as T_src tokens via self-attn encoder.
      - Decode with T_tgt learned query tokens that cross-attend to encoded source.
      - Detokenize decoded tokens to target vector.

    You can drop T_tgt to 1 (single query token) or keep small (e.g., 4) and pool.
    """
    def __init__(self,
                 d_src: int,
                 d_tgt: int,
                 d_lat: int = 512,
                 T_src: int = 8,
                 T_tgt: int = 4,
                 n_enc: int = 3,
                 n_dec: int = 3,
                 heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        assert d_lat % heads == 0, "d_lat must be divisible by heads."

        # Tokenizers
        self.src_tok = VectorTokenizer(d_src, d_lat, T_src)
        self.tgt_queries = nn.Parameter(torch.randn(T_tgt, 1, d_lat) / math.sqrt(d_lat))

        # Encoder & Decoder stacks
        self.encoder = nn.ModuleList([TransformerBlock(d_lat, heads, mlp_ratio, dropout) for _ in range(n_enc)])
        self.decoder = nn.ModuleList([XTransformerBlock(d_lat, heads, mlp_ratio, dropout) for _ in range(n_dec)])

        # Detokenizer to target space
        self.detok_tgt = VectorDetokenizer(d_lat, d_tgt, use_first_token=True)

        # Optional output refiner
        self.refine = nn.Sequential(nn.LayerNorm(d_tgt), nn.Linear(d_tgt, d_tgt))

    def forward(self, x_src):  # x_src: (B, d_src)
        # Encode source
        src_tokens = self.src_tok(x_src)  # (T_src, B, d_lat)
        for blk in self.encoder:
            src_tokens = blk(src_tokens)

        # Decode: start from learned target queries (broadcast across batch)
        B = x_src.size(0)
        tgt_tokens = self.tgt_queries.expand(-1, B, -1)  # (T_tgt, B, d_lat)
        for blk in self.decoder:
            tgt_tokens = blk(tgt_tokens, src_tokens)

        # Project to target vector
        y_hat = self.detok_tgt(tgt_tokens)  # (B, d_tgt)
        return self.refine(y_hat)

# ---------------------------
# Training & Evaluation
# ---------------------------
def train_epoch(model, dl, opt, device, w_cos=1.0, w_mse=1.0, clip=1.0):
    model.train()
    total, agg = 0, {"loss":0.0, "cos":0.0, "mse":0.0}
    for xb, yb in dl:
        xb = xb.to(device); yb = yb.to(device)
        y_hat = model(xb)
        cos = cosine_loss(y_hat, yb)
        mse = F.mse_loss(y_hat, yb)
        loss = w_cos * cos + w_mse * mse
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        bs = xb.size(0)
        total += bs
        agg["loss"] += float(loss.item()) * bs
        agg["cos"]  += float(cos.item()) * bs
        agg["mse"]  += float(mse.item()) * bs
    if total == 0:
        return {k:0.0 for k in agg}
    return {k: v/total for k, v in agg.items()}

@torch.no_grad()
def eval_epoch(model, dl, device):
    model.eval()
    total, agg = 0, {"loss":0.0, "cos":0.0, "mse":0.0}
    for xb, yb in dl:
        xb = xb.to(device); yb = yb.to(device)
        y_hat = model(xb)
        cos = cosine_loss(y_hat, yb)
        mse = F.mse_loss(y_hat, yb)
        loss = cos + mse
        bs = xb.size(0)
        total += bs
        agg["loss"] += float(loss.item()) * bs
        agg["cos"]  += float(cos.item()) * bs
        agg["mse"]  += float(mse.item()) * bs
    if total == 0:
        return {k:0.0 for k in agg}
    return {k: v/total for k, v in agg.items()}

@torch.no_grad()
def final_test_metrics(model, X_test, Y_test, device):
    model.eval()
    x = torch.tensor(X_test, dtype=torch.float32, device=device)
    y = torch.tensor(Y_test, dtype=torch.float32, device=device)
    y_pred = F.normalize(model(x), dim=-1).cpu().numpy()
    y_true = F.normalize(y, dim=-1).cpu().numpy()
    # Pairwise cosine (paired alignment)
    pair_cos = float(np.mean(np.sum(y_pred * y_true, axis=1)))
    # Retrieval @1/@5 & MRR assuming pairs (i -> i)
    sim = cosine_sim_matrix(y_pred, y_true)  # (N,N)
    order = np.argsort(-sim, axis=1)
    N = sim.shape[0]
    top1 = np.mean(order[:,0] == np.arange(N))
    top5 = np.mean([np.any(order[i,:5] == i) for i in range(N)])
    # MRR
    ranks = []
    for i in range(N):
        pos = int(np.where(order[i] == i)[0][0])
        ranks.append(pos + 1)
    mrr = float(np.mean(1.0 / np.array(ranks)))
    return {"pair_cos": pair_cos, "top1": float(top1), "top5": float(top5), "mrr": mrr}


def run(X, Y, epochs=20):
    N, d_src = X.shape[0], X.shape[1]
    d_tgt = Y.shape[1]
    print(f"N={N} | d_src={d_src} | d_tgt={d_tgt}")

    ds = PairEmbeds(X, Y, norm=True)
    train_idx, test_idx = split_indices(N, test_ratio=0.2, seed=42)

    # carve a small val from train (10%)
    val_take = max(1, int(0.1 * len(train_idx)))
    val_idx = train_idx[:val_take]
    real_train_idx = train_idx[val_take:]

    # batch sizes robust to tiny datasets
    tr_bs = max(1, min(512, len(real_train_idx)))
    va_bs = max(1, min(512, len(val_idx)))
    te_bs = max(1, min(512, len(test_idx)))

    train_dl = DataLoader(Subset(ds, real_train_idx), batch_size=tr_bs, shuffle=True, drop_last=False)
    val_dl   = DataLoader(Subset(ds, val_idx),       batch_size=va_bs, shuffle=False, drop_last=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VecMapTransformer(
        d_src=d_src, d_tgt=d_tgt,
        d_lat=512, T_src=8, T_tgt=4,
        n_enc=3, n_dec=3, heads=8, mlp_ratio=4.0, dropout=0.1
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.98), weight_decay=1e-4)

    for ep in tqdm(range(1, epochs+1), desc="Training"):
        tr = train_epoch(model, train_dl, opt, device, w_cos=1.0, w_mse=1.0, clip=1.0)
        va = eval_epoch(model, val_dl, device)
        print(f"Epoch {ep:03d} | "
              f"train loss {tr['loss']:.3f} (cos {tr['cos']:.3f} mse {tr['mse']:.3f}) | "
              f"val loss {va['loss']:.3f} (cos {va['cos']:.3f} mse {va['mse']:.3f})")

    # Final test
    X_test = ds.x[test_idx].numpy()
    Y_test = ds.y[test_idx].numpy()
    metrics = final_test_metrics(model, X_test, Y_test, device)
    print("\n=== TEST METRICS ===")
    print(json.dumps(metrics, indent=2))

    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", time.localtime(time.time()))
    if not os.path.exists("runs"): os.makedirs("runs")
    save_pth = f"runs/{current_time}"
    
    with open(f"{save_pth}vecmap_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    # torch.save(net.state_dict(), f"{save_pth}_vec2vec_xattn.pt")
    torch.save(model.state_dict(), f"runs/vecmap_transformer.pt")
    print("Saved: vecmap_test_metrics.json, vecmap_transformer.pt")


# === TEST METRICS ===
# {
#   "pair_cos": 0.8776745200157166,
#   "top1": 0.23684210526315788,
#   "top5": 0.6842105263157895,
#   "mrr": 0.4165623871388088
# }


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    return parser.parse_args()

# ---------------------------
# Main (example script)
# ---------------------------
if __name__ == "__main__":
    import pickle
    with open("datasets/eamodelset_nl2cm_embedding.pkl", "rb") as f:
        data = pickle.load(f)
    args = parse_args()
    SRC = data['NL_Serialization_Emb']
    TGT = data['CM_Serialization_Emb']
    print(f"Loaded {len(SRC)} pairs.")
    # Load once for dims & split
    X = np.stack(SRC); Y = np.stack(TGT)
    run(X, Y, epochs=500)
