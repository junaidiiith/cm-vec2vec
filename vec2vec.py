# vec2vec_xattn_train_eval.py
import os, math, numpy as np, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import time
from utils import (
    PairEmbeds, 
    l2, 
    vsp_loss, 
    set_seed, 
    cosine_loss, 
    split_indices, 
    topk_and_mrr, 
    geometry_corr,
    cosine_sim_matrix, 
)


# ---------------------------
# Transformer modules
# ---------------------------
class CrossAttnBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.q_ln = nn.LayerNorm(dim)
        self.kv_ln = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=False)
        self.ffn_ln = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.drop = nn.Dropout(dropout)
    def forward(self, x_tokens, y_tokens):
        q = self.q_ln(x_tokens)
        kv = self.kv_ln(y_tokens)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        x = x_tokens + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.ffn_ln(x)))
        return x

class SelfAttnBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=False)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.drop = nn.Dropout(dropout)
    def forward(self, t):
        qkv = self.ln1(t)
        attn_out, _ = self.attn(qkv, qkv, qkv, need_weights=False)
        x = t + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x

class VectorTokenizer(nn.Module):
    def __init__(self, d_in, d_lat, T):
        super().__init__()
        self.T = T
        self.proj = nn.Linear(d_in, T*d_lat)
        self.ln = nn.LayerNorm(d_lat)
    def forward(self, v):  # (B,d_in) -> (T,B,d_lat)
        B = v.size(0)
        t = self.proj(v).view(B, self.T, -1)
        t = self.ln(t).transpose(0,1)
        return t

class VectorDetokenizer(nn.Module):
    def __init__(self, d_lat, d_out, use_first_token=True):
        super().__init__()
        self.use_first = use_first_token
        self.proj = nn.Linear(d_lat, d_out)
    def forward(self, tokens):  # (T,B,d_lat) -> (B,d_out)
        pooled = tokens[0] if self.use_first else tokens.mean(dim=0)
        return self.proj(pooled)

class XAttnTranslator(nn.Module):
    def __init__(self, d_text, d_model, d_lat=512, T=8, heads=8, layers=2, mem_tokens=4, dropout=0.1):
        super().__init__()
        assert d_lat % heads == 0
        self.tok_text  = VectorTokenizer(d_text,  d_lat, T)
        self.tok_model = VectorTokenizer(d_model, d_lat, T)
        self.x_t2m = nn.ModuleList([CrossAttnBlock(d_lat, heads, dropout=dropout) for _ in range(layers)])
        self.x_m2t = nn.ModuleList([CrossAttnBlock(d_lat, heads, dropout=dropout) for _ in range(layers)])
        self.s_text  = nn.ModuleList([SelfAttnBlock(d_lat, heads, dropout=dropout) for _ in range(layers)])
        self.s_model = nn.ModuleList([SelfAttnBlock(d_lat, heads, dropout=dropout) for _ in range(layers)])
        self.mem_t2m = nn.Parameter(torch.randn(mem_tokens, 1, d_lat) / math.sqrt(d_lat))
        self.mem_m2t = nn.Parameter(torch.randn(mem_tokens, 1, d_lat) / math.sqrt(d_lat))
        self.detok_text  = VectorDetokenizer(d_lat, d_text)
        self.detok_model = VectorDetokenizer(d_lat, d_model)
        self.out_text  = nn.Sequential(nn.LayerNorm(d_text),  nn.Linear(d_text,  d_text))
        self.out_model = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))

    def _xstack(self, q_tok, kv_tok, blocks):
        x = q_tok
        for b in blocks: x = b(x, kv_tok)
        return x
    def _sstack(self, tok, blocks):
        x = tok
        for b in blocks: x = b(x)
        return x

    def text_to_model(self, x_vec, y_hint=None):
        x_tok = self.tok_text(x_vec)
        if y_hint is not None:
            y_tok = self.tok_model(y_hint)
            z = self._xstack(x_tok, y_tok, self.x_t2m)
        else:
            B = x_vec.size(0)
            mem = self.mem_t2m.expand(-1, B, -1)
            z = self._xstack(x_tok, mem, self.x_t2m)
            z = self._sstack(z, self.s_model)
        y_hat = self.out_model(self.detok_model(z))
        return y_hat, z

    def model_to_text(self, y_vec, x_hint=None):
        y_tok = self.tok_model(y_vec)
        if x_hint is not None:
            x_tok = self.tok_text(x_hint)
            z = self._xstack(y_tok, x_tok, self.x_m2t)
        else:
            B = y_vec.size(0)
            mem = self.mem_m2t.expand(-1, B, -1)
            z = self._xstack(y_tok, mem, self.x_m2t)
            z = self._sstack(z, self.s_text)
        x_hat = self.out_text(self.detok_text(z))
        return x_hat, z

class Vec2VecXAttn(nn.Module):
    def __init__(self, d_text, d_model, d_lat=512, T=8, heads=8, layers=2, mem_tokens=4, dropout=0.1):
        super().__init__()
        self.trans = XAttnTranslator(d_text, d_model, d_lat, T, heads, layers, mem_tokens, dropout)
    def forward(self, xb, yb):
        y_hat, _ = self.trans.text_to_model(xb, y_hint=yb)
        x_hat, _ = self.trans.model_to_text(yb, x_hint=xb)
        # memory-path recon
        y_rec, _ = self.trans.text_to_model(xb, y_hint=None)
        x_rec, _ = self.trans.model_to_text(yb, x_hint=None)
        # cycles
        x_cyc, _ = self.trans.model_to_text(y_hat, x_hint=xb)
        y_cyc, _ = self.trans.text_to_model(x_hat, y_hint=yb)
        return dict(y_hat=y_hat, x_hat=x_hat, y_rec=y_rec, x_rec=x_rec, x_cyc=x_cyc, y_cyc=y_cyc)

# ---------------------------
# Training with live validation
# ---------------------------
def train_with_val(
    train_dl, val_dl, d_text, d_model,
    d_lat=512, T=8, heads=8, layers=2, mem_tokens=4, dropout=0.1,
    epochs=20, lr=2e-4, l_rec=1.0, l_cyc=1.0, l_vsp=0.1, l_sup=1.0, device="cuda"
):
    net = Vec2VecXAttn(d_text, d_model, d_lat, T, heads, layers, mem_tokens, dropout).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    best_state_dict = None
    patience = 10
    patience_counter = 0

    def run_epoch(dl, train=True):
        if train: net.train()
        else: net.eval()
        total = 0
        agg = {"sup":0.0, "rec":0.0, "cyc":0.0, "vsp":0.0, "loss":0.0}
        
        for xb, yb in dl:
            if xb.size(0) == 0:  # Skip empty batches
                continue
                
            xb = xb.to(device); yb = yb.to(device)
            out = net(xb, yb)
            sup = cosine_loss(out["y_hat"], yb) + cosine_loss(out["x_hat"], xb)
            rec = l2(out["x_rec"], xb) + l2(out["y_rec"], yb)
            cyc = l2(out["x_cyc"], xb) + l2(out["y_cyc"], yb)
            vsp = vsp_loss(xb, out["y_hat"]) + vsp_loss(yb, out["x_hat"])
            loss = l_sup*sup + l_rec*rec + l_cyc*cyc + l_vsp*vsp
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            bs = xb.size(0)
            total += bs
            for k, v in [("sup",sup),("rec",rec),("cyc",cyc),("vsp",vsp),("loss",loss)]:
                agg[k] += float(v.item()) * bs
        if total > 0:  # Prevent division by zero
            for k in agg: agg[k] /= total
        return agg

    for ep in range(1, epochs+1):
        tr = run_epoch(train_dl, train=True)
        va = run_epoch(val_dl, train=False)
        print(f"Epoch {ep:03d} | "
              f"train loss {tr['loss']:.3f} (sup {tr['sup']:.3f} rec {tr['rec']:.3f} cyc {tr['cyc']:.3f} vsp {tr['vsp']:.3f}) | "
              f"val loss {va['loss']:.3f} (sup {va['sup']:.3f} rec {va['rec']:.3f} cyc {va['cyc']:.3f} vsp {va['vsp']:.3f})")
        
        # Learning rate scheduler and early stopping
        scheduler.step(va['loss'])
        
        # Save best model
        if va['loss'] < best_val_loss:
            best_val_loss = va['loss']
            best_state_dict = {k: v.cpu().detach() for k, v in net.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {ep}")
                break
    
    # Restore best model
    if best_state_dict is not None:
        net.load_state_dict({k: torch.tensor(v) for k, v in best_state_dict.items()})

    return net

# ---------------------------
# Evaluation on test split
# ---------------------------
@torch.no_grad()
def evaluate_on_test(net, x_test, y_test, device="cuda"):
    net.eval()
    x = torch.tensor(x_test, dtype=torch.float32, device=device)
    y = torch.tensor(y_test, dtype=torch.float32, device=device)

    # text->model (no y_hint at inference)
    y_pred, _ = net.trans.text_to_model(x, y_hint=None)
    # model->text (no x_hint)
    x_pred, _ = net.trans.model_to_text(y, x_hint=None)

    y_pred = F.normalize(y_pred, dim=-1).cpu().numpy()
    x_pred = F.normalize(x_pred, dim=-1).cpu().numpy()
    x_np   = F.normalize(x, dim=-1).cpu().numpy()
    y_np   = F.normalize(y, dim=-1).cpu().numpy()

    # Retrieval metrics (paired indices)
    sim_t2m = cosine_sim_matrix(y_pred, y_np)   # predicted models vs true models
    sim_m2t = cosine_sim_matrix(x_pred, x_np)   # predicted texts vs true texts
    r_t2m = topk_and_mrr(sim_t2m, assume_paired=True)
    r_m2t = topk_and_mrr(sim_m2t, assume_paired=True)

    # Pairwise cosine to ground truth (alignment quality)
    pair_cos_t2m = np.mean(np.sum(y_pred * y_np, axis=1))
    pair_cos_m2t = np.mean(np.sum(x_pred * x_np, axis=1))

    # Geometry preservation (neighborhood correlation) in target spaces
    geom_t2m = geometry_corr(y_np, y_pred)
    geom_m2t = geometry_corr(x_np, x_pred)

    return {
        "t2m": {"top1": r_t2m["top1"], "top5": r_t2m["top5"], "mrr": r_t2m["mrr"], "mean_rank": r_t2m["mean_rank"],
                "pair_cos": float(pair_cos_t2m), "geom_corr": float(geom_t2m)},
        "m2t": {"top1": r_m2t["top1"], "top5": r_m2t["top5"], "mrr": r_m2t["mrr"], "mean_rank": r_m2t["mean_rank"],
                "pair_cos": float(pair_cos_m2t), "geom_corr": float(geom_m2t)}
    }

def run(X, Y, epochs=20):
    set_seed(42)
    assert X.shape[0] == Y.shape[0], "Paired arrays must have same N"
    N = X.shape[0]; d_text = X.shape[1]; d_model = Y.shape[1]
    print(f"N={N} | d_text={d_text} | d_model={d_model}")

    ds_full = PairEmbeds(X, Y, norm=True)
    train_idx, test_idx = split_indices(N, test_ratio=0.2, seed=42)
    # For live validation during training, carve val out of train (e.g., 10% of train)
    tr_sz = len(train_idx)
    val_sz = max(1, int(0.1 * tr_sz))
    val_idx = train_idx[:val_sz]
    real_train_idx = train_idx[val_sz:]

    print(f"Train size: {len(real_train_idx)} | Val size: {len(val_idx)} | Test size: {len(test_idx)}")
    
    # Adjust batch size to match dataset size
    batch_size = min(64, len(real_train_idx))
    print(f"Using batch size: {batch_size}")
    
    # Recreate dataloaders with appropriate batch size and drop_last=False
    train_dl = DataLoader(Subset(ds_full, real_train_idx), batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl   = DataLoader(Subset(ds_full, val_idx), batch_size=batch_size, shuffle=False, drop_last=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = train_with_val(
        train_dl, val_dl, d_text, d_model,
        d_lat=256, T=4, heads=4, layers=2, mem_tokens=4, dropout=0.2,
        epochs=epochs, lr=3e-4, l_rec=1.0, l_cyc=1.0, l_vsp=0.2, l_sup=1.5, device=device
    )

    # Final test evaluation
    x_test = ds_full.x[test_idx].numpy()
    y_test = ds_full.y[test_idx].numpy()
    metrics = evaluate_on_test(net, x_test, y_test, device=device)
    print("\n=== TEST METRICS ===")
    for k, v in metrics.items():
        print(k, v)

    # Optionally save metrics
    # get timestamp in ddd-mm-YYYY_HH-MM-SS format
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", time.localtime(time.time()))
    
    if not os.path.exists("runs"): os.makedirs("runs")
    save_pth = f"runs/{current_time}"
    
    with open(f"{save_pth}_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    # torch.save(net.state_dict(), f"{save_pth}_vec2vec_xattn.pt")
    torch.save(net.state_dict(), f"runs/vec2vec_xattn.pt")
    print("Saved: test_metrics.json, vec2vec_xattn.pt")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    return parser.parse_args()


# ---------------------------
# Main
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
    run(X, Y, epochs=100)  

# === TEST METRICS ===
# t2m {'top1': 0.7631578947368421, 'top5': 0.9473684210526315, 'mrr': 0.8450814536340853, 'mean_rank': 1.8157894736842106, 'pair_cos': 0.6840874552726746, 'geom_corr': 0.6554960608482361}
# m2t {'top1': 0.7631578947368421, 'top5': 0.9210526315789473, 'mrr': 0.8275546321945214, 'mean_rank': 2.1578947368421053, 'pair_cos': 0.5905044674873352, 'geom_corr': 0.6112334728240967}
