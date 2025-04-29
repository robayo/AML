import random

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx

def build_nx_graph(edges: pd.DataFrame) -> nx.Graph:
    G = nx.DiGraph() 
    G.add_edges_from(edges[['src', 'dst']].itertuples(index=False, name=None))
    return G

def create_label_dict(df):
    """Create a dictionary mapping txId to class label."""
    return dict(zip(df['txId'], df['class_label']))

def guilty_walker_features(df, time_step_graphs, num_walks=500, walk_len=50, restart_prob=0.10):
    """Much faster GuiltyWalker implementation using time step information."""
    # Create a mapping of nodes to their time steps
    node_to_time = dict(zip(df['txId'], df['time_step']))
    
    # Create mapping of nodes to labels
    node_to_label = dict(zip(df['txId'], df['class_label']))
    
    # Initialize results dictionary
    feature_dict = {
        'txId': [],
        'gw_hit_ratio': [],
        'gw_min_step': [],
        'gw_avg_step': [],
        'gw_unique_illicit': [],
    }
    
    # Process each time step separately
    for time_step, G in tqdm(time_step_graphs.items(), desc='Processing time steps'):
        # Create reversed graph for backward traversal
        G_inv = G.reverse(copy=True)
        
        # Get illicit nodes for this time step
        illicit_nodes = {n for n in G.nodes() if node_to_label.get(n) == 'illicit'}
        
        # Process each node in this time step
        for v in G.nodes():
            hits = []
            step_hits = []
            illicit_encountered = set()
            
            for _ in range(num_walks):
                current = v
                walk_hit = False
                
                for step in range(1, walk_len + 1):
                    # Get predecessors (walking backward in time)
                    preds = list(G_inv.neighbors(current))
                    if not preds:
                        break
                    current = random.choice(preds)

                    # Check if current node is illicit (skipping the starting node)
                    if step > 1 and current in illicit_nodes:
                        hits.append(1)
                        step_hits.append(step)
                        illicit_encountered.add(current)
                        walk_hit = True
                        break
                    
                    # Random walk restart logic  
                    if random.random() < restart_prob:
                        current = v  # restart
                
                if not walk_hit:
                    hits.append(0)
            
            hit_ratio = np.mean(hits) if hits else 0.0
            feature_dict['txId'].append(v) 
            feature_dict['gw_hit_ratio'].append(hit_ratio)
            feature_dict['gw_unique_illicit'].append(len(illicit_encountered))
            
            if step_hits:
                feature_dict['gw_min_step'].append(min(step_hits))
                feature_dict['gw_avg_step'].append(np.mean(step_hits))
            else:
                feature_dict['gw_min_step'].append(walk_len + 1)  # infinity proxy 
                feature_dict['gw_avg_step'].append(walk_len + 1)  # infinity proxy
    
    return pd.DataFrame(feature_dict)

def build_time_step_graphs(edges, df):
    """Build separate directed graphs for each time step more efficiently."""
    # Group nodes by time_step
    time_step_nodes = df.groupby('time_step')['txId'].apply(list).to_dict()
    
    # Create graphs for each time step
    time_step_graphs = {}
    
    # Process all edges in one go
    edge_df = edges.copy()
    # Add time_step information to edges
    edge_df = edge_df.merge(df[['txId', 'time_step']], left_on='src', right_on='txId', how='inner')
    edge_df = edge_df.rename(columns={'time_step': 'src_time'})
    edge_df = edge_df.drop('txId', axis=1)
    
    # Group edges by time_step and build graphs
    for time_step, group in edge_df.groupby('src_time'):
        G = nx.DiGraph()
        # Add all nodes for this time step
        G.add_nodes_from(time_step_nodes[time_step])
        # Add edges
        G.add_edges_from(group[['src', 'dst']].values)
        time_step_graphs[time_step] = G
    
    return time_step_graphs


##############################################################################
#                 Graph Convolutional Network Embeddings                      #
##############################################################################

class GCN(nn.Module):
    def __init__(self, in_feats, hidden=64, out_feats=32, dropout=0.3):
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
def train_gcn_get_embeddings(df, time_step_graphs, feature_cols, epochs=30, lr=1e-3, device='cpu'):
    """Train semi-supervised GCN for each time step and return embeddings and models."""
    all_embeddings = []
    time_step_models = {}  # Dictionary to store trained models by time step
    
    for time_step, G in tqdm(time_step_graphs.items(), desc="GCN Embeddings"):
        if len(G.nodes()) < 10:  # Skip very small graphs
            continue
            
        # Filter dataframe for nodes in this time step
        sub_df = df[df['time_step'] == time_step]
        nodes = list(G.nodes())
        
        if len(sub_df) < 10:  # Skip if not enough data
            continue
            
        # PyG data
        pyg_graph = from_networkx(G)
        
        # Prepare feature matrix - align with node order in the graph
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Make sure all nodes are in dataframe
        nodes_in_df = [n for n in nodes if n in sub_df['txId'].values]
        
        if not nodes_in_df:
            continue
            
        # Get features for these nodes
        node_df = sub_df[sub_df['txId'].isin(nodes_in_df)]
        feat_mat = node_df.set_index('txId')[feature_cols].fillna(0).values
        
        # Create edge index for PyG
        edge_list = list(G.edges())
        src_nodes = [src for src, _ in edge_list if src in node_df['txId'].values]
        dst_nodes = [dst for _, dst in edge_list if dst in node_df['txId'].values]
        
        if not src_nodes or not dst_nodes:
            continue
            
        # Map to indices
        src_indices = [node_to_idx[src] for src in src_nodes]
        dst_indices = [node_to_idx[dst] for dst in dst_nodes]
        
        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        
        # Features and labels
        x = torch.tensor(feat_mat, dtype=torch.float32)
        
        # Map labels: illicit=1, licit=0, unknown=-1
        # For training, we'll use known labels but will also consider the graph structure
        # including unlabeled nodes
        label_map = {'illicit': 1, 'licit': 0, 'unknown': -1}
        y = node_df['class_label'].map(label_map).fillna(-1).astype(int).values
        y = torch.tensor(y, dtype=torch.long)
        
        # Setup train mask - only use nodes with known labels (illicit or licit)
        train_mask = y >= 0  # Only labeled nodes
        train_mask = torch.tensor(train_mask, dtype=torch.bool)
        
        if not train_mask.any():  # Skip if no labeled nodes
            continue
        
        # Create PyG data object
        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data = data.to(device)
        
        # Initialize model
        model = GCN(in_feats=len(feature_cols), hidden=128, out_feats=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        loss_fn = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Time step {time_step}, Epoch {epoch}: Loss {loss.item():.4f}")
        
        # Get embeddings for all nodes (including unknown)
        model.eval()
        with torch.no_grad():
            emb = model(data.x, data.edge_index).cpu().numpy()
        
        # Create embedding dataframe
        emb_df = pd.DataFrame(emb, columns=[f'gcn_{i}' for i in range(emb.shape[1])])
        emb_df['txId'] = node_df['txId'].values
        
        all_embeddings.append(emb_df)
        
        # Store the trained model
        time_step_models[time_step] = {
            'model': model.cpu(),  # Move model to CPU for storage
            'node_mapping': node_to_idx  # Save the node mapping for inference
        }
    
    # Combine all embeddings
    combined_embeddings = pd.DataFrame(columns=['txId'] + [f'gcn_{i}' for i in range(64)])
    if all_embeddings:
        combined_embeddings = pd.concat(all_embeddings, ignore_index=True)
    
    return combined_embeddings, time_step_models


def generate_embeddings(df, time_step_graphs, feature_cols, trained_models, device='cpu'):
    """Generate embeddings for new data using pre-trained models."""
    all_embeddings = []
    
    for time_step, G in tqdm(time_step_graphs.items(), desc="Generating Embeddings"):
        # Skip if we don't have a trained model for this time step
        if time_step not in trained_models:
            print(f"No trained model for time step {time_step}, skipping")
            continue
            
        if len(G.nodes()) < 10:  # Skip very small graphs
            continue
            
        # Filter dataframe for nodes in this time step
        sub_df = df[df['time_step'] == time_step]
        nodes = list(G.nodes())
        
        if len(sub_df) < 10:  # Skip if not enough data
            continue
            
        # Get the trained model for this time step
        model = trained_models[time_step]['model'].to(device)
        
        # Make sure all nodes are in dataframe
        nodes_in_df = [n for n in nodes if n in sub_df['txId'].values]
        
        if not nodes_in_df:
            continue
            
        # Get features for these nodes
        node_df = sub_df[sub_df['txId'].isin(nodes_in_df)]
        feat_mat = node_df.set_index('txId')[feature_cols].fillna(0).values
        
        # Create node index to feature row mapping for this run
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Create edge index for PyG
        edge_list = list(G.edges())
        src_nodes = [src for src, _ in edge_list if src in node_df['txId'].values]
        dst_nodes = [dst for _, dst in edge_list if dst in node_df['txId'].values]
        
        if not src_nodes or not dst_nodes:
            continue
            
        # Map to indices
        src_indices = [node_to_idx[src] for src in src_nodes]
        dst_indices = [node_to_idx[dst] for dst in dst_nodes]
        
        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        
        # Features tensor
        x = torch.tensor(feat_mat, dtype=torch.float32)
        
        # Create data object
        data = Data(x=x, edge_index=edge_index)
        data = data.to(device)
        
        # Generate embeddings
        model.eval()
        with torch.no_grad():
            emb = model(data.x, data.edge_index).cpu().numpy()
        
        # Create embedding dataframe
        emb_df = pd.DataFrame(emb, columns=[f'gcn_{i}' for i in range(emb.shape[1])])
        emb_df['txId'] = node_df['txId'].values
        
        all_embeddings.append(emb_df)
    
    # Combine all embeddings
    if all_embeddings:
        return pd.concat(all_embeddings, ignore_index=True)
    else:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=['txId'] + [f'gcn_{i}' for i in range(64)])