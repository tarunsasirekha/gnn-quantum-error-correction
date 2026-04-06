"""
Graph Representation for qLDPC Codes
Converts qLDPC codes and syndromes to PyTorch Geometric graphs
"""

import torch
import numpy as np
from torch_geometric.data import Data
from typing import Tuple, Optional, List
import networkx as nx


class TannerGraphBuilder:
    """Build Tanner graph representation from parity check matrix"""
    
    def __init__(self, H: np.ndarray, stabilizer_type: str = 'X'):
        """
        Initialize Tanner graph builder
        
        Args:
            H: Parity check matrix (m × n)
            stabilizer_type: 'X' or 'Z' to distinguish stabilizer types
        """
        self.H = H
        self.n_checks, self.n_qubits = H.shape
        self.stabilizer_type = stabilizer_type
        
        # Build edge index
        self.edge_index = self._build_edge_index()
        
    def _build_edge_index(self) -> torch.Tensor:
        """
        Build edge index for bipartite Tanner graph
        
        Returns:
            edge_index: [2, num_edges] tensor
            
        Node numbering:
            - Qubits: 0 to n_qubits-1
            - Checks: n_qubits to n_qubits+n_checks-1
        """
        edges = []
        
        for check_idx in range(self.n_checks):
            for qubit_idx in range(self.n_qubits):
                if self.H[check_idx, qubit_idx] == 1:
                    # Add bidirectional edges
                    qubit_node = qubit_idx
                    check_node = self.n_qubits + check_idx
                    
                    edges.append([qubit_node, check_node])
                    edges.append([check_node, qubit_node])
        
        if len(edges) == 0:
            # Empty graph
            return torch.zeros((2, 0), dtype=torch.long)
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index
    
    def create_graph_data(self, 
                          syndrome: np.ndarray,
                          error: Optional[np.ndarray] = None,
                          add_features: bool = True) -> Data:
        """
        Create PyTorch Geometric Data object
        
        Args:
            syndrome: Syndrome vector (length n_checks)
            error: Ground truth error vector (length n_qubits), optional
            add_features: Whether to add additional node features
            
        Returns:
            PyTorch Geometric Data object
        """
        # Node features
        qubit_features = self._create_qubit_features(add_features)
        check_features = self._create_check_features(syndrome, add_features)
        
        # Combine node features
        x = torch.cat([qubit_features, check_features], dim=0)
        
        # Create data object
        data = Data(
            x=x,
            edge_index=self.edge_index,
            num_nodes=self.n_qubits + self.n_checks
        )
        
        # Add ground truth if available
        if error is not None:
            data.y = torch.tensor(error, dtype=torch.float32)
        
        # Add metadata
        data.n_qubits = self.n_qubits
        data.n_checks = self.n_checks
        data.syndrome = torch.tensor(syndrome, dtype=torch.float32)

        # Qubit mask — True for qubit nodes, False for check nodes
        # Works correctly when PyG batches multiple graphs together
        n_total = self.n_qubits + self.n_checks
        qubit_mask = torch.zeros(n_total, dtype=torch.bool)
        qubit_mask[:self.n_qubits] = True
        data.qubit_mask = qubit_mask

        return data
    
    def _create_qubit_features(self, add_features: bool = True) -> torch.Tensor:
        """Create initial features for qubit nodes"""
        if add_features:
            # Feature: [is_qubit, degree (normalized), 0 (placeholder)]
            degrees = self.H.sum(axis=0)
            max_degree = max(degrees.max(), 1)  # normalize to [0,1]

            features = torch.zeros((self.n_qubits, 3), dtype=torch.float32)
            features[:, 0] = 1.0
            features[:, 1] = torch.tensor(degrees / max_degree, dtype=torch.float32)
            # features[:, 2] stays 0
        else:
            # Minimal features
            features = torch.zeros((self.n_qubits, 2), dtype=torch.float32)
            features[:, 0] = 1.0  # is_qubit
        
        return features
    
    def _create_check_features(self, syndrome: np.ndarray, add_features: bool = True) -> torch.Tensor:
        """Create features for check nodes from syndrome"""
        if add_features:
            # Feature: [is_check, syndrome_value, degree (normalized)]
            degrees = self.H.sum(axis=1)
            max_degree = max(degrees.max(), 1)

            features = torch.zeros((self.n_checks, 3), dtype=torch.float32)
            features[:, 0] = 0.0  # is_check (not qubit)
            features[:, 1] = torch.tensor(syndrome, dtype=torch.float32)
            features[:, 2] = torch.tensor(degrees / max_degree, dtype=torch.float32)
        else:
            features = torch.zeros((self.n_checks, 2), dtype=torch.float32)
            features[:, 0] = 0.0  # is_check
            features[:, 1] = torch.tensor(syndrome, dtype=torch.float32)
        
        return features
    
    def get_networkx_graph(self) -> nx.Graph:
        """Convert to NetworkX graph for visualization"""
        G = nx.Graph()
        
        # Add qubit nodes
        for i in range(self.n_qubits):
            G.add_node(i, node_type='qubit', label=f'q{i}')
        
        # Add check nodes
        for i in range(self.n_checks):
            node_id = self.n_qubits + i
            G.add_node(node_id, node_type='check', label=f'c{i}')
        
        # Add edges
        edge_list = self.edge_index.t().numpy()
        for edge in edge_list:
            if edge[0] < edge[1]:  # Add each edge once
                G.add_edge(edge[0], edge[1])
        
        return G


class DualGraphBuilder:
    """Build separate graphs for X and Z syndromes"""
    
    def __init__(self, H_X: np.ndarray, H_Z: np.ndarray):
        """
        Initialize dual graph builder
        
        Args:
            H_X: X-type stabilizer matrix
            H_Z: Z-type stabilizer matrix
        """
        self.graph_X = TannerGraphBuilder(H_X, stabilizer_type='X')
        self.graph_Z = TannerGraphBuilder(H_Z, stabilizer_type='Z')
    
    def create_data_pair(self,
                        syndrome_X: np.ndarray,
                        syndrome_Z: np.ndarray,
                        error_X: Optional[np.ndarray] = None,
                        error_Z: Optional[np.ndarray] = None) -> Tuple[Data, Data]:
        """
        Create pair of graphs for X and Z syndromes
        
        Args:
            syndrome_X: X syndrome (detects Z errors)
            syndrome_Z: Z syndrome (detects X errors)
            error_X: Ground truth X errors
            error_Z: Ground truth Z errors
            
        Returns:
            (data_X, data_Z): Pair of Data objects
        """
        # X syndrome detects Z errors
        data_X = self.graph_X.create_graph_data(syndrome_X, error_Z)
        
        # Z syndrome detects X errors  
        data_Z = self.graph_Z.create_graph_data(syndrome_Z, error_X)
        
        return data_X, data_Z


class BatchGraphBuilder:
    """Build batches of graphs for training"""
    
    def __init__(self, H_X: np.ndarray, H_Z: np.ndarray):
        self.dual_builder = DualGraphBuilder(H_X, H_Z)
    
    def create_batch(self, dataset: dict, indices: Optional[List[int]] = None) -> Tuple[List[Data], List[Data]]:
        """
        Create batch of graph data
        
        Args:
            dataset: Dictionary with syndrome_X, syndrome_Z, error_X, error_Z
            indices: Which samples to include (if None, use all)
            
        Returns:
            (data_X_list, data_Z_list): Lists of Data objects
        """
        if indices is None:
            indices = range(len(dataset['syndrome_X']))
        
        data_X_list = []
        data_Z_list = []
        
        for i in indices:
            data_X, data_Z = self.dual_builder.create_data_pair(
                syndrome_X=dataset['syndrome_X'][i],
                syndrome_Z=dataset['syndrome_Z'][i],
                error_X=dataset['error_X'][i] if 'error_X' in dataset else None,
                error_Z=dataset['error_Z'][i] if 'error_Z' in dataset else None
            )
            data_X_list.append(data_X)
            data_Z_list.append(data_Z)
        
        return data_X_list, data_Z_list


def visualize_tanner_graph(H: np.ndarray, syndrome: Optional[np.ndarray] = None, 
                           error: Optional[np.ndarray] = None, 
                           figsize: Tuple[int, int] = (12, 8)):
    """
    Visualize Tanner graph using matplotlib
    
    Args:
        H: Parity check matrix
        syndrome: Optional syndrome to highlight violated checks
        error: Optional error to highlight error locations
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available for visualization")
        return
    
    builder = TannerGraphBuilder(H)
    G = builder.get_networkx_graph()
    
    # Layout
    qubit_nodes = [i for i in range(builder.n_qubits)]
    check_nodes = [builder.n_qubits + i for i in range(builder.n_checks)]
    
    pos = {}
    # Qubits on top
    for i, node in enumerate(qubit_nodes):
        pos[node] = (i, 1)
    # Checks on bottom
    for i, node in enumerate(check_nodes):
        pos[node] = (i * len(qubit_nodes) / len(check_nodes), 0)
    
    plt.figure(figsize=figsize)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=qubit_nodes, 
                          node_color='lightblue', 
                          node_shape='o', 
                          node_size=500,
                          label='Qubits')
    
    check_colors = ['red' if syndrome is not None and syndrome[i] == 1 else 'lightgreen' 
                   for i in range(builder.n_checks)]
    nx.draw_networkx_nodes(G, pos, nodelist=check_nodes,
                          node_color=check_colors,
                          node_shape='s',
                          node_size=500,
                          label='Checks')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    # Draw labels
    labels = {node: G.nodes[node]['label'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title("Tanner Graph Visualization")
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from qldpc_codes import ToricCode
    from error_simulation import ErrorSimulator, NoiseModel
    
    print("=== Testing Tanner Graph Builder ===")
    
    # Create small toric code
    toric = ToricCode(L=3)
    print(f"Toric code: {toric.n_qubits} qubits, {toric.H_X.shape[0]} X-checks, {toric.H_Z.shape[0]} Z-checks")
    
    # Build graph
    graph_builder = TannerGraphBuilder(toric.H_Z)
    print(f"\nGraph structure:")
    print(f"  Nodes: {graph_builder.n_qubits} qubits + {graph_builder.n_checks} checks = {graph_builder.n_qubits + graph_builder.n_checks}")
    print(f"  Edges: {graph_builder.edge_index.shape[1]}")
    
    # Generate error and syndrome
    simulator = ErrorSimulator(toric.H_X, toric.H_Z, NoiseModel(p_depol=0.1))
    sample = simulator.generate_error_syndrome_pair()
    
    # Create graph data
    data = graph_builder.create_graph_data(
        syndrome=sample['syndrome_Z'],
        error=sample['error_X']
    )
    
    print(f"\nGraph data object:")
    print(f"  Node features shape: {data.x.shape}")
    print(f"  Edge index shape: {data.edge_index.shape}")
    print(f"  Ground truth shape: {data.y.shape}")
    print(f"  Syndrome: {data.syndrome.numpy()}")
    
    # Test dual graph builder
    print("\n=== Testing Dual Graph Builder ===")
    dual_builder = DualGraphBuilder(toric.H_X, toric.H_Z)
    
    data_X, data_Z = dual_builder.create_data_pair(
        syndrome_X=sample['syndrome_X'],
        syndrome_Z=sample['syndrome_Z'],
        error_X=sample['error_X'],
        error_Z=sample['error_Z']
    )
    
    print(f"X-syndrome graph: {data_X.num_nodes} nodes, {data_X.edge_index.shape[1]} edges")
    print(f"Z-syndrome graph: {data_Z.num_nodes} nodes, {data_Z.edge_index.shape[1]} edges")
    
    # Test batch builder
    print("\n=== Testing Batch Builder ===")
    dataset = simulator.generate_dataset(n_samples=10)
    batch_builder = BatchGraphBuilder(toric.H_X, toric.H_Z)
    
    data_X_list, data_Z_list = batch_builder.create_batch(dataset)
    print(f"Created batch with {len(data_X_list)} X-graphs and {len(data_Z_list)} Z-graphs")
    print(f"First X-graph: {data_X_list[0].num_nodes} nodes")
    
    # Test NetworkX conversion
    print("\n=== Testing NetworkX Conversion ===")
    G = graph_builder.get_networkx_graph()
    print(f"NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Node types: {set(nx.get_node_attributes(G, 'node_type').values())}")
