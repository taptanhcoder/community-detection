from __future__ import annotations

from collections import defaultdict
import random
import numpy as np
import pandas as pd


def _require_torch():
    try:
        import torch  
        return True
    except Exception:
        return False


def build_id_maps(users_final: pd.DataFrame):
    user_ids = users_final["user_id"].astype(str).tolist()
    id2idx = {u: i for i, u in enumerate(user_ids)}
    idx2id = user_ids
    return id2idx, idx2id


def build_adj_list(edges_final: pd.DataFrame, id2idx: dict):
    adj = defaultdict(list)
    for u, v in zip(edges_final["u"].astype(str), edges_final["v"].astype(str)):
        if u in id2idx and v in id2idx:
            ui = id2idx[u]; vi = id2idx[v]
            if ui != vi:
                adj[ui].append(vi)
                adj[vi].append(ui)
    return adj


def sample_neighbors(adj, nodes, sample_size: int):
    out = []
    for n in nodes.tolist():
        neigh = adj.get(int(n), [])
        if len(neigh) == 0:
            out.append([int(n)])
        elif len(neigh) <= sample_size:
            out.append(neigh)
        else:
            out.append(random.sample(neigh, sample_size))
    return out


def train_graphsage_unsup(
    edges_final: pd.DataFrame,
    users_final: pd.DataFrame,
    X_users: np.ndarray,
    hidden_dim=128,
    embed_dim=128,
    neighbor_sampling=(25, 10),
    epochs=10,
    batch_size=1024,
    num_negative=5,
    lr=1e-3,
    device=None,
    seed=42,
    train_edge_frac: float = 0.1,
    logger=None,
):
    
    if not _require_torch():
        raise RuntimeError("GraphSAGE step requires PyTorch. Install extras: pip install -e .[torch]")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if not (0.0 < train_edge_frac <= 1.0):
        raise ValueError("train_edge_frac must be in (0, 1].")

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    id2idx, idx2id = build_id_maps(users_final)
    adj = build_adj_list(edges_final, id2idx)


    pos_u = []
    pos_v = []
    for u, v in zip(edges_final["u"].astype(str), edges_final["v"].astype(str)):
        if u in id2idx and v in id2idx:
            ui = id2idx[u]; vi = id2idx[v]
            if ui != vi:
                pos_u.append(ui); pos_v.append(vi)

    pos_u = np.array(pos_u, dtype=np.int64)
    pos_v = np.array(pos_v, dtype=np.int64)
    if len(pos_u) == 0:
        raise ValueError("No positive edges after filtering. Check preprocess outputs.")


    if train_edge_frac < 1.0:
        m = len(pos_u)
        m_sub = max(1, int(m * train_edge_frac))
        idx = np.random.choice(m, size=m_sub, replace=False)
        pos_u = pos_u[idx]
        pos_v = pos_v[idx]

    N = len(idx2id)
    in_dim = int(X_users.shape[1])

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    class MeanAggregator(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.lin = nn.Linear(in_dim * 2, out_dim)

        def forward(self, self_h, neigh_h):
            return self.lin(torch.cat([self_h, neigh_h], dim=1))

    class GraphSAGE(nn.Module):
        def __init__(self, in_dim, hidden_dim, embed_dim):
            super().__init__()
            self.agg1 = MeanAggregator(in_dim, hidden_dim)
            self.agg2 = MeanAggregator(hidden_dim, embed_dim)

        def forward_batch(self, x, adj, batch_nodes, sizes=(25, 10)):
            B = batch_nodes.shape[0]
            S1, S2 = sizes

            # Layer 1
            neigh1 = sample_neighbors(adj, batch_nodes, S1)
            self_h0 = x[batch_nodes]
            neigh1_mean0 = []
            for i in range(B):
                idxs = torch.tensor(neigh1[i], device=x.device, dtype=torch.long)
                neigh1_mean0.append(x[idxs].mean(dim=0))
            neigh1_mean0 = torch.stack(neigh1_mean0, dim=0)
            h1 = F.relu(self.agg1(self_h0, neigh1_mean0))

            # Layer 2 (approx)
            neigh2 = sample_neighbors(adj, batch_nodes, S2)
            neigh2_mean1 = []
            for i in range(B):
                idxs = torch.tensor(neigh2[i], device=x.device, dtype=torch.long)
                n_nodes = idxs
                nB = n_nodes.shape[0]
                n_self0 = x[n_nodes]
                n_neigh1 = sample_neighbors(adj, n_nodes, S1)

                n_neigh_mean0 = []
                for j in range(nB):
                    j_idxs = torch.tensor(n_neigh1[j], device=x.device, dtype=torch.long)
                    n_neigh_mean0.append(x[j_idxs].mean(dim=0))
                n_neigh_mean0 = torch.stack(n_neigh_mean0, dim=0)
                n_h1 = F.relu(self.agg1(n_self0, n_neigh_mean0))
                neigh2_mean1.append(n_h1.mean(dim=0))
            neigh2_mean1 = torch.stack(neigh2_mean1, dim=0)

            z = self.agg2(h1, neigh2_mean1)
            z = F.normalize(z, p=2, dim=1)
            return z

    x = torch.tensor(X_users, dtype=torch.float32, device=device)
    model = GraphSAGE(in_dim=in_dim, hidden_dim=hidden_dim, embed_dim=embed_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))

    if logger:
        logger.info(f"[C5] device={device} | N={N} | train_pos_edges={len(pos_u)} (frac={train_edge_frac}) | in_dim={in_dim}")
        logger.info(f"[C5] epochs={epochs} | batch_size={batch_size} | neg={num_negative} | neighbor_sampling={neighbor_sampling}")

    num_batches = int(np.ceil(len(pos_u) / batch_size))

    for ep in range(1, int(epochs) + 1):
        perm = np.random.permutation(len(pos_u))
        pu = pos_u[perm]; pv = pos_v[perm]
        total_loss = 0.0
        model.train()

        for b in range(num_batches):
            s = b * batch_size
            e = min((b + 1) * batch_size, len(pu))
            bu = torch.tensor(pu[s:e], device=device)
            bv = torch.tensor(pv[s:e], device=device)

            zu = model.forward_batch(x, adj, bu, sizes=neighbor_sampling)
            zv = model.forward_batch(x, adj, bv, sizes=neighbor_sampling)

            pos_logits = (zu * zv).sum(dim=1)

            neg_v = torch.randint(low=0, high=N, size=(bu.shape[0], num_negative), device=device)
            neg_logits = []
            for j in range(num_negative):
                zvn = model.forward_batch(x, adj, neg_v[:, j], sizes=neighbor_sampling)
                neg_logits.append((zu * zvn).sum(dim=1))
            neg_logits = torch.stack(neg_logits, dim=1)

            loss_pos = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
            loss_neg = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
            loss = loss_pos + loss_neg

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())

        if logger:
            logger.info(f"[C5] epoch {ep}/{epochs} | avg_loss={(total_loss / max(1, num_batches)):.4f}")


    model.eval()
    Z = np.zeros((N, embed_dim), dtype=np.float32)
    with torch.no_grad():
        all_idx = torch.arange(N, device=device)
        bs = 2048
        for s in range(0, N, bs):
            idx = all_idx[s:s + bs]
            z = model.forward_batch(x, adj, idx, sizes=neighbor_sampling)
            Z[s:s + len(idx)] = z.detach().cpu().numpy()

    return Z, model
