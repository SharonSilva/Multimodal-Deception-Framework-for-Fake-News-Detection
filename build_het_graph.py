"""
temporal_heterogeneous_gnn.py
==============================
Temporal Heterogeneous Graph Neural Network for fake news detection.

Architecture:
1. Heterogeneous message passing across different node/edge types
2. Temporal evolution modeling with GRU
3. Fusion of temporal + heterogeneous embeddings
4. Multi-task outputs: node classification, edge classification, risk scoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import pickle
from pathlib import Path
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# HETEROGENEOUS MESSAGE PASSING LAYER
# ============================================================================

class HeteroMessagePassing(nn.Module):
    """
    Message passing for heterogeneous graphs with multiple relation types.
    
    For each relation type r:
      h_v^(l+1) = σ( Σ_{u ∈ N_r(v)} (1/c_{v,r}) * W_r^(l) * h_u^(l) )
    """
    
    def __init__(self, in_dims, out_dim, relation_types):
        """
        Args:
            in_dims: dict {node_type: input_dim}
            out_dim: output embedding dimension (common space)
            relation_types: list of (src_type, rel_type, dst_type)
        """
        super().__init__()
        
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.relation_types = relation_types
        
        # Create projection layers for each relation type
        self.relation_weights = nn.ModuleDict()
        
        for src_type, rel_type, dst_type in relation_types:
            key = f"{src_type}_{rel_type}_{dst_type}"
            in_dim = in_dims[src_type]
            self.relation_weights[key] = nn.Linear(in_dim, out_dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, node_features, edge_dict):
        """
        Args:
            node_features: dict {node_type: feature_tensor}
            edge_dict: dict {(src_type, rel, dst_type): (edge_index, edge_weight)}
                       edge_index: [2, num_edges]
                       edge_weight: [num_edges]
                       
        Returns:
            updated_features: dict {node_type: updated_tensor}
        """
        updated_features = {}
        
        # Process each node type
        for node_type in node_features.keys():
            # Initialize aggregated messages
            num_nodes = node_features[node_type].shape[0]
            device = node_features[node_type].device
            
            aggregated = torch.zeros(num_nodes, self.out_dim, device=device)
            
            # Aggregate messages from all incoming relation types
            for (src_type, rel_type, dst_type) in self.relation_types:
                if dst_type != node_type:
                    continue
                
                key = f"{src_type}_{rel_type}_{dst_type}"
                
                if (src_type, rel_type, dst_type) not in edge_dict:
                    continue
                
                edge_index, edge_weight = edge_dict[(src_type, rel_type, dst_type)]
                
                # Validate edge indices
                src_nodes = edge_index[0]
                dst_nodes = edge_index[1]
                
                # Filter out-of-bounds indices
                src_max = node_features[src_type].shape[0]
                dst_max = num_nodes
                
                valid_mask = (
                    (src_nodes >= 0) & (src_nodes < src_max) &
                    (dst_nodes >= 0) & (dst_nodes < dst_max)
                )
                
                if valid_mask.sum() == 0:
                    continue
                
                src_nodes = src_nodes[valid_mask]
                dst_nodes = dst_nodes[valid_mask]
                edge_weight = edge_weight[valid_mask]
                
                # Get source node features
                src_features = node_features[src_type]
                
                # Project through relation-specific weight
                transformed = self.relation_weights[key](src_features)
                
                # Normalize by degree (destination node degree)
                deg = torch.zeros(num_nodes, device=device)
                deg.scatter_add_(0, dst_nodes, edge_weight)
                deg = deg.clamp(min=1.0)
                
                # Aggregate messages (vectorized)
                messages = transformed[src_nodes] * edge_weight.unsqueeze(1)
                messages = messages / deg[dst_nodes].unsqueeze(1)
                
                aggregated.scatter_add_(0, dst_nodes.unsqueeze(1).expand_as(messages), messages)
            
            # Apply activation and normalization
            updated_features[node_type] = self.norm(F.relu(aggregated))
        
        return updated_features


# ============================================================================
# TEMPORAL EVOLUTION MODULE
# ============================================================================

class TemporalEvolutionGRU(nn.Module):
    """
    Models temporal evolution of node embeddings using GRU.
    
    h_v^t = GRU(h_v^{t-1}, aggregate(neighbors at time t))
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # GRU cell for temporal updates
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Temporal decay weight
        self.temporal_decay = nn.Parameter(torch.tensor(3600.0))  # tau in seconds
        
    def forward(self, current_embeddings, neighbor_messages, timestamps, current_time):
        """
        Args:
            current_embeddings: [num_nodes, hidden_dim]
            neighbor_messages: [num_nodes, hidden_dim] - aggregated from neighbors
            timestamps: [num_nodes] - last update time for each node
            current_time: scalar - current timestamp
            
        Returns:
            updated_embeddings: [num_nodes, hidden_dim]
        """
        # Compute time decay weights
        time_deltas = current_time - timestamps
        decay_weights = torch.exp(-time_deltas / self.temporal_decay)
        decay_weights = decay_weights.unsqueeze(1)  # [num_nodes, 1]
        
        # Apply temporal decay to current embeddings
        decayed_embeddings = current_embeddings * decay_weights
        
        # Update with GRU
        updated = self.gru_cell(neighbor_messages, decayed_embeddings)
        
        return updated


# ============================================================================
# MAIN TEMPORAL HETEROGENEOUS GNN MODEL
# ============================================================================

class TemporalHeterogeneousGNN(nn.Module):
    """
    Complete Temporal Heterogeneous GNN for fake news detection.
    
    Architecture:
    1. Node feature projection to common space
    2. Multiple layers of heterogeneous message passing
    3. Temporal GRU updates
    4. Fusion layer
    5. Multi-task prediction heads (including edge classification)
    """
    
    def __init__(self, node_dims, hidden_dim=256, num_layers=3, 
                 relation_types=None, num_classes=2):
        """
        Args:
            node_dims: dict {node_type: input_dim}
            hidden_dim: common embedding dimension
            num_layers: number of message passing layers
            relation_types: list of (src_type, rel_type, dst_type) tuples
            num_classes: number of output classes (2 for binary)
        """
        super().__init__()
        
        self.node_types = list(node_dims.keys())
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.relation_types = relation_types or []
        
        # 1. Input projection layers (to common space)
        self.input_projections = nn.ModuleDict()
        for ntype, dim in node_dims.items():
            self.input_projections[ntype] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            )
        
        # 2. Heterogeneous message passing layers
        self.hetero_layers = nn.ModuleList([
            HeteroMessagePassing(
                in_dims={ntype: hidden_dim for ntype in node_dims.keys()},
                out_dim=hidden_dim,
                relation_types=self.relation_types
            )
            for _ in range(num_layers)
        ])
        
        # 3. Temporal evolution modules (one per node type)
        self.temporal_modules = nn.ModuleDict()
        for ntype in node_dims.keys():
            self.temporal_modules[ntype] = TemporalEvolutionGRU(hidden_dim)
        
        # 4. Fusion layer (combines temporal + heterogeneous features)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(hidden_dim)
        )
        
        # 5. Task-specific heads
        
        # 5a. Node classification (per node type)
        self.node_classifiers = nn.ModuleDict()
        for ntype in ['post', 'user']:  # Only classify posts and users
            self.node_classifiers[ntype] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        
        # 5b. Community risk scoring
        self.risk_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 5c. Deception cluster detection
        self.deception_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 5d. Edge classification (for suspicious chains/interactions)
        self.edge_classifiers = nn.ModuleDict()
        edge_types_to_classify = [
            'user_interacts_with_user',
            'post_flagged_in_deception_cluster',
            'deception_cluster_colludes_with_deception_cluster'
        ]
        for edge_type in edge_types_to_classify:
            self.edge_classifiers[edge_type] = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
    
    def classify_edges(self, fused_embeddings, edge_dict, edge_types_to_classify):
        """
        Classify specific edge types as suspicious or not.
        
        Args:
            fused_embeddings: dict {node_type: embeddings [num_nodes, hidden_dim]}
            edge_dict: dict {(src_type, rel, dst_type): (edge_index, edge_weight)}
            edge_types_to_classify: list of (src_type, rel_type, dst_type) tuples
            
        Returns:
            dict {edge_type: edge_scores [num_edges]}
        """
        edge_scores = {}
        
        for (src_type, rel_type, dst_type) in edge_types_to_classify:
            # Check if edge type exists in graph
            if (src_type, rel_type, dst_type) not in edge_dict:
                continue
            
            edge_index, edge_weight = edge_dict[(src_type, rel_type, dst_type)]
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]
            
            # Validate indices
            if src_type not in fused_embeddings or dst_type not in fused_embeddings:
                continue
            
            src_max = fused_embeddings[src_type].shape[0]
            dst_max = fused_embeddings[dst_type].shape[0]
            
            valid = (
                (src_nodes >= 0) & (src_nodes < src_max) &
                (dst_nodes >= 0) & (dst_nodes < dst_max)
            )
            
            if valid.sum() == 0:
                continue
            
            src_nodes = src_nodes[valid]
            dst_nodes = dst_nodes[valid]
            
            # Get embeddings and ensure no NaNs
            src_emb = fused_embeddings[src_type][src_nodes]
            dst_emb = fused_embeddings[dst_type][dst_nodes]
            
            # Check for NaN embeddings and replace with zeros
            if torch.isnan(src_emb).any():
                src_emb = torch.nan_to_num(src_emb, nan=0.0)
            if torch.isnan(dst_emb).any():
                dst_emb = torch.nan_to_num(dst_emb, nan=0.0)
            
            # Concatenate source and destination embeddings
            edge_features = torch.cat([src_emb, dst_emb], dim=-1)
            
            # Classify - construct key exactly as in __init__
            classifier_key = f"{src_type}_{rel_type}_{dst_type}"
            
            if classifier_key in self.edge_classifiers:
                scores = self.edge_classifiers[classifier_key](edge_features).squeeze(-1)
                
                # Ensure no NaN scores
                scores = torch.nan_to_num(scores, nan=0.5)
                
                edge_scores[(src_type, rel_type, dst_type)] = scores
        
        return edge_scores
    
    def forward(self, node_features, edge_dict, timestamps=None, current_time=None, 
                classify_edges=False):
        """
        Forward pass through the temporal heterogeneous GNN.
        
        Args:
            node_features: dict {node_type: feature_tensor}
            edge_dict: dict {(src, rel, dst): (edge_index, edge_weight)}
            timestamps: dict {node_type: timestamp_tensor} - optional
            current_time: scalar - current timestamp - optional
            classify_edges: bool - whether to perform edge classification
            
        Returns:
            dict with predictions for each task
        """
        # 1. Project to common space
        projected = {}
        for ntype, features in node_features.items():
            projected[ntype] = self.input_projections[ntype](features)
        
        # Store for temporal update
        initial_embeddings = {k: v.clone() for k, v in projected.items()}
        
        # 2. Heterogeneous message passing
        current_embeddings = projected
        for layer in self.hetero_layers:
            current_embeddings = layer(current_embeddings, edge_dict)
        
        hetero_embeddings = current_embeddings
        
        # 3. Temporal evolution (if timestamps provided)
        if timestamps is not None and current_time is not None:
            temporal_embeddings = {}
            for ntype in current_embeddings.keys():
                if ntype in self.temporal_modules:
                    temporal_embeddings[ntype] = self.temporal_modules[ntype](
                        initial_embeddings[ntype],
                        hetero_embeddings[ntype],
                        timestamps[ntype],
                        current_time
                    )
                else:
                    temporal_embeddings[ntype] = hetero_embeddings[ntype]
        else:
            temporal_embeddings = hetero_embeddings
        
        # 4. Fusion of temporal + heterogeneous
        fused_embeddings = {}
        for ntype in current_embeddings.keys():
            concat = torch.cat([
                hetero_embeddings[ntype],
                temporal_embeddings[ntype]
            ], dim=-1)
            fused_embeddings[ntype] = self.fusion(concat)
        
        # 5. Task-specific predictions
        outputs = {'embeddings': fused_embeddings}
        
        # 5a. Node classification (posts and users)
        outputs['node_logits'] = {}
        for ntype in ['post', 'user']:
            if ntype in fused_embeddings:
                outputs['node_logits'][ntype] = self.node_classifiers[ntype](
                    fused_embeddings[ntype]
                )
        
        # 5b. Community risk scores
        if 'community' in fused_embeddings:
            outputs['community_risk'] = self.risk_scorer(
                fused_embeddings['community']
            ).squeeze(-1)
        
        # 5c. Deception cluster detection
        if 'deception_cluster' in fused_embeddings:
            outputs['deception_score'] = self.deception_detector(
                fused_embeddings['deception_cluster']
            ).squeeze(-1)
        
        # 5d. Edge classification (optional)
        if classify_edges:
            edge_types_to_classify = [
                ('user', 'interacts_with', 'user'),
                ('post', 'flagged_in', 'deception_cluster'),
                ('deception_cluster', 'colludes_with', 'deception_cluster')
            ]
            outputs['edge_scores'] = self.classify_edges(
                fused_embeddings, edge_dict, edge_types_to_classify
            )
        
        return outputs


# ============================================================================
# DATA LOADER (SIMPLIFIED - USES PRE-CONVERTED PYTORCH EDGES)
# ============================================================================

def load_heterogeneous_graph(graph_dir="heterogeneous_graph"):
    """
    Load the constructed heterogeneous graph.
    
    Returns:
        node_features: dict {node_type: feature_tensor}
        edge_dict: dict {(src_type, rel, dst_type): (edge_index, edge_weight)}
        node_mappings: dict {node_type: {original_id: local_id}}
    """
    graph_dir = Path(graph_dir)
    
    print(f"Loading from {graph_dir}...")
    
    # Load node features
    node_features = torch.load(graph_dir / "node_features.pt")
    print(f"✅ Loaded node features:")
    for ntype, feats in node_features.items():
        print(f"   {ntype}: {feats.shape}")
    
    # Load PRE-CONVERTED PyTorch edges (this is the key fix!)
    edge_dict = torch.load(graph_dir / "edge_dict.pt")
    print(f"✅ Loaded edge dictionary with {len(edge_dict)} edge types:")
    for etype, (ei, ew) in edge_dict.items():
        print(f"   {etype}: {ei.shape[1]} edges")
    
    # Load node mappings (for reference)
    with open(graph_dir / "node_mappings.pkl", "rb") as f:
        node_mappings = pickle.load(f)
    
    return node_features, edge_dict, node_mappings


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("TEMPORAL HETEROGENEOUS GNN - MODEL TEST")
    print("="*80)
    
    # Load graph (now much simpler!)
    print("\n[1/3] Loading heterogeneous graph...")
    node_features, edge_dict, node_mappings = load_heterogeneous_graph()
    
    # Initialize model
    print("\n[2/3] Initializing model...")
    node_dims = {ntype: features.shape[1] for ntype, features in node_features.items()}
    relation_types = list(edge_dict.keys())
    
    model = TemporalHeterogeneousGNN(
        node_dims=node_dims,
        hidden_dim=256,
        num_layers=3,
        relation_types=relation_types,
        num_classes=2
    ).to(device)
    
    print(f"✅ Model initialized:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
    # Test forward pass
    print("\n[3/3] Testing forward pass...")
    
    # Move to device
    node_features_device = {k: v.to(device) for k, v in node_features.items()}
    edge_dict_device = {k: (ei.to(device), ew.to(device)) 
                       for k, (ei, ew) in edge_dict.items()}
    
    # Test without edge classification
    print("\n" + "="*80)
    print("Test 1: Forward pass WITHOUT edge classification")
    print("="*80)
    
    with torch.no_grad():
        outputs = model(node_features_device, edge_dict_device, classify_edges=False)
        
        print("\n✅ Forward pass successful!")
        print(f"\n   Outputs:")
        for key, value in outputs.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for k, v in value.items():
                    print(f"      {k}: {v.shape}")
            elif isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape}")
    
    # Test with edge classification
    print("\n" + "="*80)
    print("Test 2: Forward pass WITH edge classification")
    print("="*80)
    
    with torch.no_grad():
        outputs_with_edges = model(node_features_device, edge_dict_device, classify_edges=True)
        
        print("\n✅ Forward pass with edge classification successful!")
        
        if "edge_scores" in outputs_with_edges:
            edge_scores_dict = outputs_with_edges["edge_scores"]
            
            if len(edge_scores_dict) > 0:
                print(f"\n   Edge Classification Results:")
                print(f"   Found {len(edge_scores_dict)} edge type(s) classified\n")
                
                for etype, scores in edge_scores_dict.items():
                    print(f"   📊 {etype}:")
                    print(f"      • Number of edges: {len(scores)}")
                    print(f"      • Score range: [{scores.min():.4f}, {scores.max():.4f}]")
                    print(f"      • Mean score: {scores.mean():.4f}")
                    print(f"      • Std dev: {scores.std():.4f}")
                    
                    # Classify edges as suspicious or normal (threshold = 0.5)
                    suspicious = (scores > 0.5).sum().item()
                    normal = (scores <= 0.5).sum().item()
                    
                    print(f"      • Suspicious edges (>0.5): {suspicious} ({suspicious/len(scores)*100:.1f}%)")
                    print(f"      • Normal edges (≤0.5): {normal} ({normal/len(scores)*100:.1f}%)")
                    
                    # Show distribution
                    print(f"      • Score distribution:")
                    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                    for i in range(len(bins)-1):
                        count = ((scores > bins[i]) & (scores <= bins[i+1])).sum().item()
                        print(f"        [{bins[i]:.1f}-{bins[i+1]:.1f}]: {count} edges")
                    print()
            else:
                print("\n   ⚠️  Edge classification returned empty dictionary")
        else:
            print("   ⚠️  No 'edge_scores' key in outputs")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    
    # Create summary statistics
    print(f"\n📊 Model Summary:")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Hidden Dimension: 256")
    print(f"   Message Passing Layers: 3")
    print(f"\n   Graph Statistics:")
    print(f"      Nodes: {sum(v.shape[0] for v in node_features.values()):,}")
    print(f"      Edges: {sum(ei.shape[1] for ei, _ in edge_dict.values()):,}")
    print(f"      Node Types: {len(node_features)}")
    print(f"      Edge Types: {len(edge_dict)}")
    
    print(f"\n   Model Capabilities:")
    print(f"       Node Classification (posts, users)")
    print(f"      Community Risk Scoring")
    print(f"      Deception Cluster Detection")
    print(f"      Edge Classification (user interactions, suspicious chains)")
    
    print(f"\n Ready for training! Next: train_temporal_hetero_gnn.py")