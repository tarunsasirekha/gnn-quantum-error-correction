"""
qLDPC Code Generators
Creates various quantum LDPC codes including toric codes, hypergraph product codes, etc.
"""

import numpy as np
from typing import Tuple, Optional
import scipy.sparse as sp


class ToricCode:
    """Generate toric code parity check matrices"""
    
    def __init__(self, L: int):
        """
        Initialize L×L toric code
        
        Args:
            L: Lattice dimension (L×L grid)
        """
        self.L = L
        self.n_qubits = 2 * L * L  # L² horizontal + L² vertical edges
        self.n_stars = L * L  # vertex stabilizers
        self.n_plaquettes = L * L  # face stabilizers
        
        self.H_X, self.H_Z = self._build_stabilizers()
        self.H = np.vstack([self.H_X, self.H_Z])
        
    def _build_stabilizers(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build star (X-type) and plaquette (Z-type) stabilizers"""
        L = self.L
        
        # H_X: Star stabilizers (vertex operators)
        H_X = np.zeros((self.n_stars, self.n_qubits), dtype=np.int8)
        
        for i in range(L):
            for j in range(L):
                vertex_idx = i * L + j
                
                # Four edges touching this vertex (periodic boundaries)
                # Horizontal edge above
                e_up = i * L + j
                # Horizontal edge below
                e_down = ((i + 1) % L) * L + j
                # Vertical edge to left
                e_left = L * L + i * L + j
                # Vertical edge to right
                e_right = L * L + i * L + ((j + 1) % L)
                
                H_X[vertex_idx, [e_up, e_down, e_left, e_right]] = 1
        
        # H_Z: Plaquette stabilizers (face operators)
        H_Z = np.zeros((self.n_plaquettes, self.n_qubits), dtype=np.int8)
        
        for i in range(L):
            for j in range(L):
                plaq_idx = i * L + j
                
                # Four edges around this plaquette (periodic boundaries)
                e_top = i * L + j
                e_bottom = ((i + 1) % L) * L + j
                e_left = L * L + i * L + j
                e_right = L * L + i * L + ((j + 1) % L)
                
                H_Z[plaq_idx, [e_top, e_bottom, e_left, e_right]] = 1
        
        return H_X, H_Z
    
    def get_distance(self) -> int:
        """Code distance (minimum weight of logical operator)"""
        return self.L
    
    def get_logical_operators(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get logical X and Z operators
        Returns: (logical_X, logical_Z) each shape (2, n_qubits)
        """
        L = self.L
        logical_X = np.zeros((2, self.n_qubits), dtype=np.int8)
        logical_Z = np.zeros((2, self.n_qubits), dtype=np.int8)
        
        # Logical X_1: horizontal loop of X operators
        for j in range(L):
            logical_X[0, j] = 1  # Top row horizontal edges
        
        # Logical X_2: vertical loop of X operators
        for i in range(L):
            logical_X[1, L * L + i * L] = 1  # Left column vertical edges
        
        # Logical Z_1: vertical loop of Z operators (dual to X_1)
        for i in range(L):
            logical_Z[0, L * L + i * L] = 1
        
        # Logical Z_2: horizontal loop of Z operators (dual to X_2)
        for j in range(L):
            logical_Z[1, j] = 1
        
        return logical_X, logical_Z


class HypergraphProductCode:
    """Generate hypergraph product codes from two classical codes"""
    
    def __init__(self, H1: np.ndarray, H2: np.ndarray):
        """
        Create hypergraph product code from two classical parity check matrices
        
        Args:
            H1: First classical code parity check matrix (m1 × n1)
            H2: Second classical code parity check matrix (m2 × n2)
        """
        self.H1 = H1
        self.H2 = H2
        
        m1, n1 = H1.shape
        m2, n2 = H2.shape
        
        self.n_qubits = m1 * n2 + m2 * n1
        
        # Build X and Z stabilizers
        self.H_X = self._build_X_stabilizers()
        self.H_Z = self._build_Z_stabilizers()
        self.H = np.vstack([self.H_X, self.H_Z])
    
    def _build_X_stabilizers(self) -> np.ndarray:
        """Build X-type stabilizers"""
        m1, n1 = self.H1.shape
        m2, n2 = self.H2.shape
        
        # H_X has m1*m2 rows
        H_X = np.zeros((m1 * m2, self.n_qubits), dtype=np.int8)
        
        # First block: H1 ⊗ I_{n2}
        for i in range(m1):
            for j in range(m2):
                row_idx = i * m2 + j
                for k in range(n1):
                    if self.H1[i, k] == 1:
                        for l in range(n2):
                            col_idx = k * n2 + l
                            H_X[row_idx, col_idx] = 1
        
        # Second block: I_{m1} ⊗ H2^T
        for i in range(m1):
            for j in range(m2):
                row_idx = i * m2 + j
                for k in range(n2):
                    if self.H2[j, k] == 1:
                        col_idx = m1 * n2 + i * n2 + k
                        H_X[row_idx, col_idx] = 1
        
        return H_X
    
    def _build_Z_stabilizers(self) -> np.ndarray:
        """Build Z-type stabilizers"""
        m1, n1 = self.H1.shape
        m2, n2 = self.H2.shape
        
        # H_Z has n1*n2 rows
        H_Z = np.zeros((n1 * n2, self.n_qubits), dtype=np.int8)
        
        # First block: I_{n1} ⊗ H2
        for i in range(n1):
            for j in range(n2):
                row_idx = i * n2 + j
                for k in range(m2):
                    if self.H2[k, j] == 1:
                        col_idx = i * n2 + k
                        H_Z[row_idx, col_idx] = 1
        
        # Second block: H1^T ⊗ I_{m2}
        for i in range(n1):
            for j in range(n2):
                row_idx = i * n2 + j
                for k in range(m1):
                    if self.H1[k, i] == 1:
                        for l in range(m2):
                            col_idx = m1 * n2 + k * n2 + l
                            H_Z[row_idx, col_idx] = 1
        
        return H_Z


class RandomRegularCode:
    """Generate random regular qLDPC codes"""
    
    def __init__(self, n: int, d_v: int, d_c: int, seed: Optional[int] = None):
        """
        Generate random regular LDPC code
        
        Args:
            n: Number of qubits
            d_v: Variable node degree (checks per qubit)
            d_c: Check node degree (qubits per check)
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.n_qubits = n
        self.d_v = d_v
        self.d_c = d_c
        
        # Number of checks
        assert (n * d_v) % d_c == 0, "Invalid parameters: n*d_v must be divisible by d_c"
        self.n_checks = (n * d_v) // d_c
        
        self.H_X = self._generate_regular_matrix()
        self.H_Z = self._generate_regular_matrix()
        self.H = np.vstack([self.H_X, self.H_Z])
    
    def _generate_regular_matrix(self) -> np.ndarray:
        """Generate random regular parity check matrix"""
        H = np.zeros((self.n_checks, self.n_qubits), dtype=np.int8)
        
        # Create list of (check, qubit) pairs
        edges = []
        for v in range(self.n_qubits):
            for _ in range(self.d_v):
                edges.append(v)
        
        # Shuffle and assign to checks
        np.random.shuffle(edges)
        
        for c in range(self.n_checks):
            for i in range(self.d_c):
                v = edges[c * self.d_c + i]
                H[c, v] = 1
        
        return H


def get_classical_hamming_code() -> np.ndarray:
    """Get [7,4,3] Hamming code parity check matrix"""
    H = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ], dtype=np.int8)
    return H


def get_classical_repetition_code(n: int) -> np.ndarray:
    """Get repetition code parity check matrix"""
    H = np.zeros((n-1, n), dtype=np.int8)
    for i in range(n-1):
        H[i, i] = 1
        H[i, i+1] = 1
    return H


if __name__ == "__main__":
    # Test toric code
    print("=== Toric Code L=4 ===")
    toric = ToricCode(L=4)
    print(f"Qubits: {toric.n_qubits}")
    print(f"Stabilizers: {toric.H.shape[0]}")
    print(f"Distance: {toric.get_distance()}")
    print(f"H_X shape: {toric.H_X.shape}")
    print(f"H_Z shape: {toric.H_Z.shape}")
    
    # Check stabilizer properties
    print(f"H_X row weight (should be 4): {toric.H_X.sum(axis=1)[:5]}")
    print(f"H_X col weight (should be 2): {toric.H_X.sum(axis=0)[:5]}")
    
    # Test hypergraph product code
    print("\n=== Hypergraph Product Code ===")
    H_classical = get_classical_repetition_code(3)
    hp_code = HypergraphProductCode(H_classical, H_classical)
    print(f"Qubits: {hp_code.n_qubits}")
    print(f"X stabilizers: {hp_code.H_X.shape[0]}")
    print(f"Z stabilizers: {hp_code.H_Z.shape[0]}")
    
    # Test random regular code
    print("\n=== Random Regular Code ===")
    random_code = RandomRegularCode(n=20, d_v=3, d_c=4, seed=42)
    print(f"Qubits: {random_code.n_qubits}")
    print(f"Checks: {random_code.n_checks}")
    print(f"H shape: {random_code.H.shape}")
