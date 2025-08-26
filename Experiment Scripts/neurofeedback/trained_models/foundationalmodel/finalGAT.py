#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool
from torch_geometric.seed import seed_everything
from torch_geometric.utils import add_self_loops
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG
# ---------------------------
server_dir     = '/scratch/uceerjp/'
cache_path     = os.path.join(server_dir, 'Full_plv_graph_dataset.pkl')
results_path   = os.path.join(server_dir, 'trainvaltest_results.pkl')
fig_path       = os.path.join(server_dir, 'train_val_test_curves.png')
best_model_path = os.path.join(server_dir, 'best_gat_model.pt')
use_subset     = True  # Set this to True to only use a subset of electrodes

num_epochs   = 50
batch_size   = 32
lr           = 0.001
h1 = 32
h2 = 16
h3 = 8
heads   = 7
dropout = 0.1

seed_everything(12345)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------
# ELECTRODE SUBSET SETUP
# ---------------------------

# Stieger dataset electrode labels (known order)
stieger_electrodes = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7',
    'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'
]

# Your headset channel layout (order), including only electrodes you have
headset_electrodes = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7',
    'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2'
]

# Subset indices in Stieger order that match the headset layout
subset_indices = [stieger_electrodes.index(e) for e in headset_electrodes if e in stieger_electrodes]


# ---------------------------
# SIMPLE 3-LAYER GAT MODEL
# ---------------------------
class SimpleGAT(nn.Module):
    def __init__(self, in_channels, h1, h2, h3, num_heads, dropout):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, h1, heads=num_heads, concat=True, dropout=dropout)
        self.gn1   = GraphNorm(h1 * num_heads)
        self.conv2 = GATv2Conv(h1 * num_heads, h2, heads=num_heads, concat=True, dropout=dropout)
        self.gn2   = GraphNorm(h2 * num_heads)
        self.conv3 = GATv2Conv(h2 * num_heads, h3, heads=num_heads, concat=False, dropout=dropout)
        self.gn3   = GraphNorm(h3)
        self.lin   = nn.Linear(h3, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.gn1(self.conv1(x, edge_index)))
        x = F.relu(self.gn2(self.conv2(x, edge_index)))
        x = F.relu(self.gn3(self.conv3(x, edge_index)))
        x = global_mean_pool(x, batch)
        return self.lin(x)

# ---------------------------
# THRESHOLDING + SUBSETTING
# ---------------------------
def preprocess_graph(data, topk_percent=0.5, epsilon=1e-6): 
    plv = data.x.clone().detach()  # shape [N, N]

    # Apply subset if enabled
    if use_subset:
        plv = plv[subset_indices, :][:, subset_indices]

    num_nodes = plv.shape[0]
    plv.fill_diagonal_(0.0)

    # Apply -log(1 - PLV + e) transformation
    plv = -torch.log(1.0 - plv + epsilon)

    # Edge thresholding on transformed PLV
    triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
    edge_weights = plv[triu_indices[0], triu_indices[1]]

    k = int(len(edge_weights) * topk_percent)
    topk_indices = torch.topk(edge_weights, k=k, sorted=False).indices

    row = triu_indices[0][topk_indices]
    col = triu_indices[1][topk_indices]

    edge_index = torch.cat([
        torch.stack([row, col], dim=0),
        torch.stack([col, row], dim=0)
    ], dim=1)

    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    data.x = plv
    data.edge_index = edge_index

    if hasattr(data, 'edge_attr'):
        del data.edge_attr

    return data


# ---------------------------
# LOAD & SPLIT
# ---------------------------
with open(cache_path, 'rb') as f:
    all_graphs, subject_ids = pickle.load(f)[:2]

by_subj = {}
for g in all_graphs:
    g = preprocess_graph(g)
    sid = int(g.subject) if hasattr(g.subject, 'item') else g.subject
    by_subj.setdefault(sid, []).append(g)

train_graphs, val_graphs, test_graphs = [], [], []
for subj, graphs in by_subj.items():
    n = len(graphs)
    i80 = int(0.8 * n)
    i90 = int(0.9 * n)
    train_graphs.extend(graphs[:i80])
    val_graphs.extend(graphs[i80:i90])
    test_graphs.extend(graphs[i90:])

print(f"Total graphs -> train: {len(train_graphs)}, val: {len(val_graphs)}, test: {len(test_graphs)}")
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_graphs,  batch_size=batch_size, shuffle=False)

# ---------------------------
# TRAIN / VAL / TEST LOOP
# ---------------------------
in_feats = train_graphs[0].x.shape[1]
model     = SimpleGAT(in_channels=in_feats, h1=h1, h2=h2, h3=h3, num_heads=heads, dropout=dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

train_accs, val_accs, test_accs = [], [], []
best_val_acc = 0.0
best_state   = None

for epoch in range(1, num_epochs + 1):
    model.train()
    correct = total = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        correct += (logits.argmax(dim=1) == batch.y).sum().item()
        total += batch.num_graphs
    train_acc = correct / total
    train_accs.append(train_acc)

    model.eval()
    def evaluate(loader):
        correct, total = 0, 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                preds = model(batch).argmax(dim=1)
                correct += (preds == batch.y).sum().item()
                total += batch.num_graphs
        return correct / total if total > 0 else 0.0

    val_acc  = evaluate(val_loader)
    test_acc = evaluate(test_loader)
    val_accs.append(val_acc)
    test_accs.append(test_acc)

    scheduler.step(val_acc)
    print(f"Epoch {epoch}/{num_epochs} - Train Acc: {train_acc:.2%}, Val Acc: {val_acc:.2%}, Test Acc: {test_acc:.2%}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state   = model.state_dict()

# ---------------------------
# SAVE BEST MODEL
# ---------------------------
if best_state:
    model.load_state_dict(best_state)
    torch.save(best_state, best_model_path)
    print(f"Best model weights saved to {best_model_path}")

# ---------------------------
# SAVE RESULTS
# ---------------------------
plt.figure()
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs,   label='Val Acc')
plt.plot(test_accs,  label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.savefig(fig_path)
print(f"Saved plot to {fig_path}")

with open(results_path, 'wb') as f:
    pickle.dump({
        'train_accs': train_accs,
        'val_accs':   val_accs,
        'test_accs':  test_accs,
        'best_val_acc': best_val_acc
    }, f)
print(f"Results saved to {results_path}")
