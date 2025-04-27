
import argparse
import pathlib
import random
import time
from typing import List, Dict

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import lightgbm as lgb

def build_nx_graph(edges: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    G.add_edges_from(edges[['src', 'dst']].itertuples(index=False, name=None))
    return G


##############################################################################
#                        GuiltyWalker Feature Set                             #
##############################################################################

def guilty_walker_features(G: nx.Graph,
                           labels: Dict[int, str],
                           num_walks: int = 16,
                           walk_len: int = 20,
                           restart_prob: float = 0.15) -> pd.DataFrame:
    """Compute GuiltyWalker‑style random‑walk statistics for every node.

    For each node v, launch `num_walks` random walks. Any visit to an
    `illicit` node marks the walk as suspicious. We record:
      – gw_hit_ratio: proportion of walks that hit an illicit node
      – gw_min_step: min steps until first illicit hit (∞ if none)
      – gw_avg_step: average step index of first illicit hit
    """
    illicit_nodes = {n for n, lbl in labels.items() if lbl == ' illicit'}
    feature_dict = {
        'txId': [],
        'gw_hit_ratio': [],
        'gw_min_step': [],
        'gw_avg_step': [],
    }

    for v in tqdm(G.nodes(), desc='GuiltyWalker'):
        hits = []
        min_steps = []
        for _ in range(num_walks):
            current = v
            for step in range(1, walk_len + 1):
                if current in illicit_nodes:
                    hits.append(1)
                    min_steps.append(step)
                    break
                if random.random() < restart_prob:
                    current = v  # restart
                else:
                    nbrs = list(G.neighbors(current))
                    if not nbrs:
                        break
                    current = random.choice(nbrs)
            else:
                hits.append(0)
        hit_ratio = np.mean(hits)
        feature_dict['txId'].append(v)
        feature_dict['gw_hit_ratio'].append(hit_ratio)
        if min_steps:
            feature_dict['gw_min_step'].append(min(min_steps))
            feature_dict['gw_avg_step'].append(np.mean(min_steps))
        else:
            feature_dict['gw_min_step'].append(walk_len + 1)
            feature_dict['gw_avg_step'].append(walk_len + 1)
    return pd.DataFrame(feature_dict)


##############################################################################
#                 Graph Convolutional Network Embeddings                      #
##############################################################################

class GCN(nn.Module):
    def __init__(self, in_feats: int, hidden: int = 64, out_feats: int = 32,
                 dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden)
        self.conv2 = GCNConv(hidden, out_feats)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


def train_gcn_get_embeddings(G: nx.Graph, df: pd.DataFrame,
                             feature_cols: List[str],
                             epochs: int = 30,
                             lr: float = 1e‑3,
                             device: str = 'cpu') -> pd.DataFrame:
    """Train semi‑supervised GCN and return hidden embeddings."""
    # PyG data
    pyg_graph = from_networkx(G)
    # Align features order with node index
    feat_mat = df.set_index('txId').loc[pyg_graph['x_idx'] if 'x_idx' in pyg_graph else G.nodes()].fillna(0)[feature_cols].values
    pyg_graph.x = torch.tensor(feat_mat, dtype=torch.float32)
    # Labels: 1=illicit, 0=licit, ‑1=unknown
    label_map = {' illict': 1, ' licit': 0, 'unknown': -1}
    y = df.set_index('txId')['class_label'].map(label_map).fillna(-1).astype(int).values
    pyg_graph.y = torch.tensor(y, dtype=torch.long)
    train_mask = pyg_graph.y >= 0  # only labelled nodes
    pyg_graph.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    pyg_graph = pyg_graph.to(device)

    model = GCN(in_feats=len(feature_cols), hidden=128, out_feats=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e‑4)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(pyg_graph.x, pyg_graph.edge_index)
        loss = loss_fn(out[pyg_graph.train_mask], pyg_graph.y[pyg_graph.train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f'Epoch {epoch:02d}: Loss {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        emb = model(pyg_graph.x, pyg_graph.edge_index).cpu().numpy()

    emb_df = pd.DataFrame(emb, columns=[f'gcn_{i}' for i in range(emb.shape[1])])
    emb_df['txId'] = df.set_index('txId').loc[pyg_graph['x_idx'] if 'x_idx' in pyg_graph else G.nodes()].index
    return emb_df
