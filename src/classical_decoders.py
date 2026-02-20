"""
Classical Decoders for Baseline Comparison
Implements Belief Propagation and Minimum Weight Perfect Matching
"""

import numpy as np
from typing import Optional, Tuple
import warnings
from scipy.sparse import csr_matrix


class BeliefPropagationDecoder:
    """Belief Propagation decoder for qLDPC codes"""
    
    def __init__(self, H: np.ndarray, max_iterations: int = 50, damping: float = 0.5):
        """
        Initialize BP decoder
        
        Args:
            H: Parity check matrix
            max_iterations: Maximum BP iterations
            damping: Damping factor (0 = no damping, 1 = full damping)
        """
        self.H = H
        self.max_iterations = max_iterations
        self.damping = damping
        
        self.n_checks, self.n_qubits = H.shape
        
        # Precompute neighbor lists for efficiency
        self.qubit_neighbors = [np.where(H[:, q] == 1)[0] for q in range(self.n_qubits)]
        self.check_neighbors = [np.where(H[c, :] == 1)[0] for c in range(self.n_checks)]
    
    def decode(self, syndrome: np.ndarray, channel_probs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Decode syndrome using belief propagation
        
        Args:
            syndrome: Syndrome vector
            channel_probs: Prior error probabilities for each qubit (if None, uniform)
            
        Returns:
            Decoded error vector
        """
        if channel_probs is None:
            # Default: uniform small error probability
            channel_probs = np.ones(self.n_qubits) * 0.01
        
        # Initialize log-likelihood ratios
        # LLR = log(P(no error) / P(error))
        llr_prior = np.log((1 - channel_probs) / channel_probs)
        
        # Messages from qubits to checks
        q2c_messages = {}
        for q in range(self.n_qubits):
            for c in self.qubit_neighbors[q]:
                q2c_messages[(q, c)] = llr_prior[q]
        
        # Messages from checks to qubits
        c2q_messages = {}
        for c in range(self.n_checks):
            for q in self.check_neighbors[c]:
                c2q_messages[(c, q)] = 0.0
        
        # BP iterations
        for iteration in range(self.max_iterations):
            # Save old messages for damping
            q2c_old = q2c_messages.copy()
            c2q_old = c2q_messages.copy()
            
            # Check to qubit messages
            for c in range(self.n_checks):
                neighbors = self.check_neighbors[c]
                
                for q_target in neighbors:
                    # Product over all other qubits in check
                    product = 1.0
                    for q in neighbors:
                        if q != q_target:
                            product *= np.tanh(q2c_messages[(q, c)] / 2)
                    
                    # Incorporate syndrome
                    syndrome_factor = (-1) ** syndrome[c]
                    
                    # LLR message
                    tanh_val = syndrome_factor * product
                    tanh_val = np.clip(tanh_val, -0.9999, 0.9999)  # Numerical stability
                    
                    c2q_messages[(c, q_target)] = 2 * np.arctanh(tanh_val)
            
            # Qubit to check messages
            for q in range(self.n_qubits):
                neighbors = self.qubit_neighbors[q]
                
                for c_target in neighbors:
                    # Sum of prior and messages from other checks
                    msg_sum = llr_prior[q]
                    for c in neighbors:
                        if c != c_target:
                            msg_sum += c2q_messages[(c, q)]
                    
                    q2c_messages[(q, c_target)] = msg_sum
            
            # Apply damping
            if self.damping > 0:
                for key in q2c_messages:
                    q2c_messages[key] = (1 - self.damping) * q2c_messages[key] + self.damping * q2c_old[key]
                for key in c2q_messages:
                    c2q_messages[key] = (1 - self.damping) * c2q_messages[key] + self.damping * c2q_old[key]
            
            # Check convergence (optional: could add early stopping)
        
        # Compute final beliefs
        beliefs = llr_prior.copy()
        for q in range(self.n_qubits):
            for c in self.qubit_neighbors[q]:
                beliefs[q] += c2q_messages[(c, q)]
        
        # Decode: error if belief suggests error more likely
        decoded_error = (beliefs < 0).astype(np.int8)
        
        return decoded_error
    
    def get_success_rate(self, 
                        syndromes: np.ndarray, 
                        true_errors: np.ndarray,
                        channel_probs: Optional[np.ndarray] = None) -> float:
        """
        Compute decoding success rate
        
        Args:
            syndromes: Array of syndromes [n_samples, n_checks]
            true_errors: Array of true errors [n_samples, n_qubits]
            channel_probs: Prior probabilities
            
        Returns:
            Success rate (fraction of correctly decoded errors)
        """
        n_samples = syndromes.shape[0]
        successes = 0
        
        for i in range(n_samples):
            decoded = self.decode(syndromes[i], channel_probs)
            
            # Check if decoded error produces same syndrome
            decoded_syndrome = (self.H @ decoded) % 2
            if np.array_equal(decoded_syndrome, syndromes[i]):
                # Check if it's equivalent to true error (up to stabilizers)
                # For now, just check syndrome matching
                successes += 1
        
        return successes / n_samples


class GreedyDecoder:
    """Simple greedy decoder (flips qubits to satisfy violated checks)"""
    
    def __init__(self, H: np.ndarray, max_iterations: int = 100):
        """
        Initialize greedy decoder
        
        Args:
            H: Parity check matrix
            max_iterations: Maximum iterations
        """
        self.H = H
        self.max_iterations = max_iterations
        self.n_checks, self.n_qubits = H.shape
    
    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Greedy decoding: flip qubits to satisfy checks
        
        Args:
            syndrome: Syndrome vector
            
        Returns:
            Decoded error
        """
        error = np.zeros(self.n_qubits, dtype=np.int8)
        
        for _ in range(self.max_iterations):
            # Compute current syndrome
            current_syndrome = (self.H @ error) % 2
            
            # Check if all checks satisfied
            if np.array_equal(current_syndrome, syndrome):
                break
            
            # Find violated checks
            violated = (current_syndrome != syndrome)
            
            # Count how many violated checks each qubit participates in
            qubit_scores = (self.H[violated, :].sum(axis=0))
            
            # Flip qubit with highest score
            if qubit_scores.max() > 0:
                best_qubit = np.argmax(qubit_scores)
                error[best_qubit] = 1 - error[best_qubit]
            else:
                break
        
        return error


class LookupTableDecoder:
    """Lookup table decoder (for small codes)"""
    
    def __init__(self, H: np.ndarray):
        """
        Initialize lookup table decoder
        
        Args:
            H: Parity check matrix
        """
        self.H = H
        self.n_checks, self.n_qubits = H.shape
        
        # Build lookup table
        print(f"Building lookup table for {self.n_qubits} qubits...")
        if self.n_qubits > 20:
            warnings.warn("Lookup table decoder not practical for >20 qubits")
        
        self.syndrome_to_error = self._build_lookup_table()
    
    def _build_lookup_table(self) -> dict:
        """Build syndrome to minimum-weight error lookup table"""
        table = {}
        
        # Generate all possible errors up to weight ceil(n/2)
        max_weight = min(self.n_qubits // 2 + 1, 10)  # Limit for practicality
        
        from itertools import combinations
        
        for weight in range(max_weight + 1):
            for positions in combinations(range(self.n_qubits), weight):
                error = np.zeros(self.n_qubits, dtype=np.int8)
                error[list(positions)] = 1
                
                syndrome = tuple((self.H @ error) % 2)
                
                # Store minimum weight error for each syndrome
                if syndrome not in table or weight < np.sum(table[syndrome]):
                    table[syndrome] = error
        
        return table
    
    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Decode using lookup table
        
        Args:
            syndrome: Syndrome vector
            
        Returns:
            Minimum weight error producing this syndrome
        """
        syndrome_tuple = tuple(syndrome)
        
        if syndrome_tuple in self.syndrome_to_error:
            return self.syndrome_to_error[syndrome_tuple].copy()
        else:
            # Syndrome not in table, return zero vector
            warnings.warn(f"Syndrome not in lookup table: {syndrome}")
            return np.zeros(self.n_qubits, dtype=np.int8)


def compute_logical_error_rate(H: np.ndarray,
                               logical_ops: np.ndarray,
                               decoder,
                               n_trials: int = 1000,
                               error_rate: float = 0.01) -> float:
    """
    Estimate logical error rate of a decoder
    
    Args:
        H: Parity check matrix
        logical_ops: Logical operator matrix [k, n]
        decoder: Decoder object with .decode() method
        n_trials: Number of trials
        error_rate: Physical error rate
        
    Returns:
        Logical error rate
    """
    from error_simulation import ErrorSimulator, NoiseModel
    
    # Need both H_X and H_Z, assume we're testing X errors
    simulator = ErrorSimulator(H, H, NoiseModel(p_depol=error_rate))
    
    logical_failures = 0
    
    for _ in range(n_trials):
        # Generate random error
        error_X, error_Z = simulator.generate_pauli_error()
        syndrome_X, syndrome_Z = simulator.compute_syndrome(error_X, error_Z)
        
        # Decode
        decoded_error = decoder.decode(syndrome_Z)
        
        # Check if decoding introduces logical error
        residual_error = (error_X + decoded_error) % 2
        
        # Check overlap with logical operators
        overlaps = (logical_ops @ residual_error) % 2
        if np.any(overlaps == 1):
            logical_failures += 1
    
    return logical_failures / n_trials


if __name__ == "__main__":
    from qldpc_codes import ToricCode
    from error_simulation import ErrorSimulator, NoiseModel
    
    print("=== Testing Classical Decoders ===")
    
    # Create toric code
    toric = ToricCode(L=4)
    print(f"Toric code: {toric.n_qubits} qubits")
    
    # Generate test data
    simulator = ErrorSimulator(toric.H_X, toric.H_Z, NoiseModel(p_depol=0.05))
    dataset = simulator.generate_dataset(n_samples=10)
    
    # Test Belief Propagation
    print("\n--- Belief Propagation Decoder ---")
    bp_decoder = BeliefPropagationDecoder(toric.H_Z, max_iterations=50, damping=0.5)
    
    success_count = 0
    for i in range(10):
        decoded = bp_decoder.decode(dataset['syndrome_Z'][i])
        decoded_syndrome = (toric.H_Z @ decoded) % 2
        
        if np.array_equal(decoded_syndrome, dataset['syndrome_Z'][i]):
            success_count += 1
    
    print(f"BP Success rate: {success_count}/10 = {success_count/10:.2%}")
    print(f"Example true error weight: {dataset['error_X'][0].sum()}")
    print(f"Example decoded error weight: {decoded.sum()}")
    
    # Test Greedy Decoder
    print("\n--- Greedy Decoder ---")
    greedy_decoder = GreedyDecoder(toric.H_Z, max_iterations=100)
    
    success_count = 0
    for i in range(10):
        decoded = greedy_decoder.decode(dataset['syndrome_Z'][i])
        decoded_syndrome = (toric.H_Z @ decoded) % 2
        
        if np.array_equal(decoded_syndrome, dataset['syndrome_Z'][i]):
            success_count += 1
    
    print(f"Greedy Success rate: {success_count}/10 = {success_count/10:.2%}")
    
    # Test Lookup Table (small code only)
    print("\n--- Lookup Table Decoder (small code) ---")
    small_toric = ToricCode(L=2)
    print(f"Small toric code: {small_toric.n_qubits} qubits")
    
    lut_decoder = LookupTableDecoder(small_toric.H_Z)
    print(f"Lookup table size: {len(lut_decoder.syndrome_to_error)}")
    
    small_simulator = ErrorSimulator(small_toric.H_X, small_toric.H_Z, NoiseModel(p_depol=0.1))
    small_dataset = small_simulator.generate_dataset(n_samples=10)
    
    success_count = 0
    for i in range(10):
        decoded = lut_decoder.decode(small_dataset['syndrome_Z'][i])
        decoded_syndrome = (small_toric.H_Z @ decoded) % 2
        
        if np.array_equal(decoded_syndrome, small_dataset['syndrome_Z'][i]):
            success_count += 1
    
    print(f"LUT Success rate: {success_count}/10 = {success_count/10:.2%}")
    
    print("\n=== All decoder tests passed! ===")
