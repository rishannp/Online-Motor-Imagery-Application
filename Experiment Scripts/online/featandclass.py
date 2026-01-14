import numpy as np
import scipy.signal as sig
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool
from torch_geometric.utils import add_self_loops

from config import TOPK_PERCENT, EPSILON


# -----------------------
# Fast PLV (vectorized)
# -----------------------
def compute_plv_fast(seg: np.ndarray) -> np.ndarray:
    """
    seg: [T, C] float
    returns plv: [C, C] in [0,1]
    """
    analytic = sig.hilbert(seg, axis=0)
    phase = np.angle(analytic)  # [T, C]
    E = np.exp(1j * phase)      # [T, C]
    T = max(1, E.shape[0])

    # mean(exp(i(φj-φi))) = (1/T) * conj(E)^T @ E
    M = (np.conj(E).T @ E) / float(T)     # [C, C] complex
    plv = np.abs(M).astype(np.float32)
    np.fill_diagonal(plv, 1.0)
    return plv


def plv_to_graph(plv: np.ndarray, topk_percent: float = TOPK_PERCENT, epsilon: float = EPSILON) -> Data:
    """
    Node features: transformed PLV matrix (C,C)  <-- matches your training snippet
    Edges: top-k of transformed weights (undirected) + self loops
    """
    # transform like your training: -log(1 - PLV + eps)
    W = -np.log(1.0 - plv + epsilon).astype(np.float32)
    np.fill_diagonal(W, 0.0)

    C = W.shape[0]
    triu = np.triu_indices(C, k=1)
    w = W[triu]
    k = max(1, int(round(w.size * float(topk_percent))))
    top_idx = np.argpartition(w, -k)[-k:]

    rows = triu[0][top_idx]
    cols = triu[1][top_idx]

    ei = np.hstack([
        np.stack([rows, cols], axis=0),
        np.stack([cols, rows], axis=0),
    ])
    edge_index = torch.tensor(ei, dtype=torch.long)
    edge_index, _ = add_self_loops(edge_index, num_nodes=C)

    x = torch.from_numpy(W)  # [C, C]
    return Data(x=x, edge_index=edge_index)


# -----------------------
# Model (match your foundation)
# -----------------------
class SimpleGAT(nn.Module):
    def __init__(self, in_ch, h1, h2, h3, heads, dropout=0.1):
        super().__init__()
        self.conv1 = GATv2Conv(in_ch,    h1, heads=heads, concat=True,  dropout=dropout)
        self.gn1   = GraphNorm(h1 * heads)
        self.conv2 = GATv2Conv(h1*heads, h2, heads=heads, concat=True,  dropout=dropout)
        self.gn2   = GraphNorm(h2 * heads)
        self.conv3 = GATv2Conv(h2*heads, h3, heads=heads, concat=False, dropout=dropout)
        self.gn3   = GraphNorm(h3)
        self.lin   = nn.Linear(h3, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.gn1(self.conv1(x, edge_index)))
        x = F.relu(self.gn2(self.conv2(x, edge_index)))
        x = F.relu(self.gn3(self.conv3(x, edge_index)))
        x = global_mean_pool(x, batch)
        return self.lin(x)


def infer_dims_from_state_dict(sd: dict):
    """
    Same trick you used: infer heads + hidden dims from saved weights.
    """
    heads = sd['conv1.att'].shape[1]
    h1    = sd['conv1.att'].shape[2]
    h2    = sd['conv2.att'].shape[2]
    h3    = sd['conv3.att'].shape[2]
    in_ch = sd['conv1.lin_l.weight'].shape[1]
    return in_ch, h1, h2, h3, heads


class GATPredictor:
    def __init__(self, foundation_pt: str, device: str):
        self.device = torch.device(device)
        sd = torch.load(foundation_pt, map_location=self.device)
        in_ch, h1, h2, h3, heads = infer_dims_from_state_dict(sd)
        self.model = SimpleGAT(in_ch, h1, h2, h3, heads).to(self.device)
        self.model.load_state_dict(sd)
        self.model.eval()

    @torch.no_grad()
    def predict_lr(self, window_58: np.ndarray):
        """
        window_58: [T, 58]
        returns:
          cmd: 0 (Left) or 1 (Right)
          conf: softmax max prob
        """
        plv = compute_plv_fast(window_58)         # [58,58]
        g = plv_to_graph(plv)                     # x=[58,58], edge_index
        g.y = torch.tensor([0], dtype=torch.long) # dummy
        g.batch = torch.zeros(g.x.shape[0], dtype=torch.long)

        g = g.to(self.device)
        logits = self.model(g)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        cmd = int(torch.argmax(probs).item())
        conf = float(torch.max(probs).item())
        return cmd, conf
