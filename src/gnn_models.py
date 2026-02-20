"""
Graph Neural Network Decoder Models
GNN architectures for quantum error correction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Tuple
import math


class GNNDecoderLayer(MessagePassing):
    """Single GNN layer for quantum error correction"""
    
    def __init__(self, in_channels: int, out_channels: int, aggr: str = 'add'):
        """
        Initialize GNN layer
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            aggr: Aggregation method ('add', 'mean', 'max')
        """
        super().__init__(aggr=aggr)
        
        # Message network
        self.msg_nn = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # Update network
        self.update_nn = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
    def forward(self, x, edge_index):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
        """
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        """
        Create messages from neighbor nodes
        
        Args:
            x_j: Neighbor node features [num_edges, in_channels]
        """
        return self.msg_nn(x_j)
    
    def update(self, aggr_out, x):
        """
        Update node features
        
        Args:
            aggr_out: Aggregated messages [num_nodes, out_channels]
            x: Current node features [num_nodes, in_channels]
        """
        # Concatenate aggregated messages with current features
        combined = torch.cat([x, aggr_out], dim=-1)
        return self.update_nn(combined)


class AttentionGNNLayer(MessagePassing):
    """GNN layer with attention mechanism"""
    
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4):
        """
        Initialize attention GNN layer
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension per head
            heads: Number of attention heads
        """
        super().__init__(aggr='add')
        
        self.heads = heads
        self.out_channels = out_channels
        
        # Attention parameters
        self.lin_key = nn.Linear(in_channels, heads * out_channels)
        self.lin_query = nn.Linear(in_channels, heads * out_channels)
        self.lin_value = nn.Linear(in_channels, heads * out_channels)
        
        # Output projection
        self.lin_out = nn.Linear(heads * out_channels, heads * out_channels)
        
    def forward(self, x, edge_index):
        """Forward pass with attention"""
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j, edge_index, size_i):
        """
        Compute attention-weighted messages
        
        Args:
            x_i: Target node features [num_edges, in_channels]
            x_j: Source node features [num_edges, in_channels]
        """
        # Compute queries, keys, values
        query = self.lin_query(x_i).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x_j).view(-1, self.heads, self.out_channels)
        value = self.lin_value(x_j).view(-1, self.heads, self.out_channels)
        
        # Attention scores
        alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = F.softmax(alpha, dim=1)
        
        # Weighted values
        out = alpha.unsqueeze(-1) * value
        return out.view(-1, self.heads * self.out_channels)
    
    def update(self, aggr_out):
        """Project aggregated features"""
        return self.lin_out(aggr_out)


class GNNDecoder(nn.Module):
    """Complete GNN decoder for qLDPC codes"""
    
    def __init__(self, 
                 input_dim: int = 3,
                 hidden_dim: int = 64,
                 output_dim: int = 1,
                 num_layers: int = 5,
                 use_attention: bool = False,
                 dropout: float = 0.1):
        """
        Initialize GNN decoder
        
        Args:
            input_dim: Input node feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (1 for binary classification)
            num_layers: Number of GNN layers
            use_attention: Whether to use attention mechanism
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            if use_attention:
                layer = AttentionGNNLayer(hidden_dim, hidden_dim // 4, heads=4)
            else:
                layer = GNNDecoderLayer(hidden_dim, hidden_dim)
            self.gnn_layers.append(layer)
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output projection (for qubit nodes only)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, data):
        """
        Forward pass
        
        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [num_nodes, input_dim]
                - edge_index: Edge indices [2, num_edges]
                - n_qubits: Number of qubit nodes
        
        Returns:
            predictions: Error predictions for qubit nodes [n_qubits, output_dim]
        """
        x, edge_index = data.x, data.edge_index
        n_qubits = data.n_qubits
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # Message passing rounds
        for i, gnn_layer in enumerate(self.gnn_layers):
            # GNN layer
            x_new = gnn_layer(x, edge_index)
            
            # Batch normalization
            x_new = self.batch_norms[i](x_new)
            
            # Residual connection
            if x_new.shape == x.shape:
                x_new = x_new + x
            
            # Activation and dropout
            x = F.relu(x_new)
            x = self.dropout(x)
        
        # Extract qubit node features
        qubit_features = x[:n_qubits]
        
        # Output projection
        output = self.output_proj(qubit_features)
        
        return output


class DualGNNDecoder(nn.Module):
    """Dual decoder for X and Z errors simultaneously"""
    
    def __init__(self,
                 input_dim: int = 3,
                 hidden_dim: int = 64,
                 num_layers: int = 5,
                 use_attention: bool = False,
                 dropout: float = 0.1,
                 shared_encoder: bool = False):
        """
        Initialize dual decoder
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
            use_attention: Use attention mechanism
            dropout: Dropout rate
            shared_encoder: Whether to share encoder between X and Z decoders
        """
        super().__init__()
        
        self.shared_encoder = shared_encoder
        
        if shared_encoder:
            # Single shared encoder
            self.encoder = GNNDecoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=1,
                num_layers=num_layers,
                use_attention=use_attention,
                dropout=dropout
            )
        else:
            # Separate encoders for X and Z
            self.decoder_X = GNNDecoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=1,
                num_layers=num_layers,
                use_attention=use_attention,
                dropout=dropout
            )
            
            self.decoder_Z = GNNDecoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=1,
                num_layers=num_layers,
                use_attention=use_attention,
                dropout=dropout
            )
    
    def forward(self, data_X, data_Z):
        """
        Forward pass for both X and Z syndromes
        
        Args:
            data_X: Data for X syndrome (detects Z errors)
            data_Z: Data for Z syndrome (detects X errors)
            
        Returns:
            (pred_X, pred_Z): Predictions for X and Z errors
        """
        if self.shared_encoder:
            pred_Z = self.encoder(data_X)  # X syndrome detects Z errors
            pred_X = self.encoder(data_Z)  # Z syndrome detects X errors
        else:
            pred_Z = self.decoder_X(data_X)
            pred_X = self.decoder_Z(data_Z)
        
        return pred_X, pred_Z


class ResidualGNNDecoder(nn.Module):
    """GNN decoder with strong residual connections"""
    
    def __init__(self,
                 input_dim: int = 3,
                 hidden_dim: int = 64,
                 num_layers: int = 8,
                 dropout: float = 0.1):
        """
        Initialize residual GNN decoder
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension (must be consistent for residuals)
            num_layers: Number of layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_layers = num_layers
        
        # Input projection to hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers with same dimension for residuals
        self.gnn_layers = nn.ModuleList([
            GNNDecoderLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Layer normalization instead of batch norm
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, data):
        """Forward pass with residual connections"""
        x, edge_index = data.x, data.edge_index
        n_qubits = data.n_qubits
        
        # Project to hidden dimension
        x = self.input_proj(x)
        x = F.relu(x)
        
        # Residual blocks
        for i in range(self.num_layers):
            # Store residual
            residual = x
            
            # GNN layer
            x = self.gnn_layers[i](x, edge_index)
            
            # Add residual
            x = x + residual
            
            # Layer norm
            x = self.layer_norms[i](x)
            
            # Activation and dropout
            x = F.relu(x)
            x = self.dropout(x)
        
        # Output for qubits only
        qubit_features = x[:n_qubits]
        output = self.output_proj(qubit_features)
        
        return output


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from qldpc_codes import ToricCode
    from error_simulation import ErrorSimulator, NoiseModel
    from graph_representation import TannerGraphBuilder
    
    print("=== Testing GNN Decoder ===")
    
    # Create toric code
    toric = ToricCode(L=4)
    
    # Generate error and syndrome
    simulator = ErrorSimulator(toric.H_X, toric.H_Z, NoiseModel(p_depol=0.05))
    sample = simulator.generate_error_syndrome_pair()
    
    # Create graph
    graph_builder = TannerGraphBuilder(toric.H_Z)
    data = graph_builder.create_graph_data(
        syndrome=sample['syndrome_Z'],
        error=sample['error_X']
    )
    
    # Test basic decoder
    print("\n--- Basic GNN Decoder ---")
    decoder = GNNDecoder(input_dim=3, hidden_dim=64, num_layers=5)
    print(f"Parameters: {count_parameters(decoder):,}")
    
    with torch.no_grad():
        output = decoder(data)
    print(f"Input shape: {data.x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test attention decoder
    print("\n--- Attention GNN Decoder ---")
    attention_decoder = GNNDecoder(
        input_dim=3,
        hidden_dim=64,
        num_layers=5,
        use_attention=True
    )
    print(f"Parameters: {count_parameters(attention_decoder):,}")
    
    with torch.no_grad():
        output_attn = attention_decoder(data)
    print(f"Output shape: {output_attn.shape}")
    
    # Test residual decoder
    print("\n--- Residual GNN Decoder ---")
    residual_decoder = ResidualGNNDecoder(
        input_dim=3,
        hidden_dim=64,
        num_layers=8
    )
    print(f"Parameters: {count_parameters(residual_decoder):,}")
    
    with torch.no_grad():
        output_res = residual_decoder(data)
    print(f"Output shape: {output_res.shape}")
    
    # Test dual decoder
    print("\n--- Dual GNN Decoder ---")
    from graph_representation import DualGraphBuilder
    
    dual_builder = DualGraphBuilder(toric.H_X, toric.H_Z)
    data_X, data_Z = dual_builder.create_data_pair(
        syndrome_X=sample['syndrome_X'],
        syndrome_Z=sample['syndrome_Z'],
        error_X=sample['error_X'],
        error_Z=sample['error_Z']
    )
    
    dual_decoder = DualGNNDecoder(
        input_dim=3,
        hidden_dim=64,
        num_layers=5,
        shared_encoder=False
    )
    print(f"Parameters: {count_parameters(dual_decoder):,}")
    
    with torch.no_grad():
        pred_X, pred_Z = dual_decoder(data_X, data_Z)
    print(f"X prediction shape: {pred_X.shape}")
    print(f"Z prediction shape: {pred_Z.shape}")
    
    print("\n=== All tests passed! ===")
