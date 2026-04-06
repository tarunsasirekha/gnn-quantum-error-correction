"""
Error Simulation and Syndrome Generation
Simulates quantum errors and generates training data for GNN decoder
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class NoiseModel:
    """Quantum noise model parameters"""
    p_x: float = 0.01  # X error probability
    p_y: float = 0.01  # Y error probability  
    p_z: float = 0.01  # Z error probability
    p_depol: float = 0.0  # Depolarizing channel probability
    
    def get_total_error_rate(self) -> float:
        """Total error probability"""
        if self.p_depol > 0:
            return self.p_depol
        return self.p_x + self.p_y + self.p_z


class ErrorSimulator:
    """Simulate quantum errors and generate syndromes"""
    
    def __init__(self, H_X: np.ndarray, H_Z: np.ndarray, noise_model: Optional[NoiseModel] = None):
        """
        Initialize error simulator
        
        Args:
            H_X: X-type stabilizer matrix
            H_Z: Z-type stabilizer matrix
            noise_model: Noise model (default: depolarizing with p=0.01)
        """
        self.H_X = H_X
        self.H_Z = H_Z
        self.n_qubits = H_X.shape[1]
        
        if noise_model is None:
            noise_model = NoiseModel(p_depol=0.01)
        self.noise_model = noise_model
    
    def generate_pauli_error(self, error_rate: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random Pauli error
        
        Args:
            error_rate: Override default error rate
            
        Returns:
            (error_X, error_Z): Binary error vectors
        """
        if error_rate is None:
            error_rate = self.noise_model.get_total_error_rate()
        
        error_X = np.zeros(self.n_qubits, dtype=np.int8)
        error_Z = np.zeros(self.n_qubits, dtype=np.int8)
        
        if self.noise_model.p_depol > 0:
            # Depolarizing channel: each qubit has p_depol chance of I, X, Y, or Z
            for i in range(self.n_qubits):
                if np.random.random() < error_rate:
                    error_type = np.random.randint(1, 4)  # 1=X, 2=Y, 3=Z
                    if error_type == 1:  # X
                        error_X[i] = 1
                    elif error_type == 2:  # Y = iXZ
                        error_X[i] = 1
                        error_Z[i] = 1
                    else:  # Z
                        error_Z[i] = 1
        else:
            # Independent X, Y, Z errors
            for i in range(self.n_qubits):
                rand = np.random.random()
                if rand < self.noise_model.p_x:
                    error_X[i] = 1
                elif rand < self.noise_model.p_x + self.noise_model.p_y:
                    error_X[i] = 1
                    error_Z[i] = 1
                elif rand < self.noise_model.p_x + self.noise_model.p_y + self.noise_model.p_z:
                    error_Z[i] = 1
        
        return error_X, error_Z
    
    def compute_syndrome(self, error_X: np.ndarray, error_Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute syndrome from errors
        
        X errors detected by Z stabilizers: syndrome_Z = H_Z @ error_X (mod 2)
        Z errors detected by X stabilizers: syndrome_X = H_X @ error_Z (mod 2)
        
        Args:
            error_X: X component of error
            error_Z: Z component of error
            
        Returns:
            (syndrome_X, syndrome_Z): Binary syndrome vectors
        """
        syndrome_X = (self.H_X @ error_Z) % 2
        syndrome_Z = (self.H_Z @ error_X) % 2
        
        return syndrome_X.astype(np.int8), syndrome_Z.astype(np.int8)
    
    def generate_error_syndrome_pair(self, error_rate: Optional[float] = None) -> Dict:
        """
        Generate one training sample: error + syndrome
        
        Returns:
            Dictionary with keys: error_X, error_Z, syndrome_X, syndrome_Z
        """
        error_X, error_Z = self.generate_pauli_error(error_rate)
        syndrome_X, syndrome_Z = self.compute_syndrome(error_X, error_Z)
        
        return {
            'error_X': error_X,
            'error_Z': error_Z,
            'syndrome_X': syndrome_X,
            'syndrome_Z': syndrome_Z,
            'error_rate': error_rate if error_rate else self.noise_model.get_total_error_rate()
        }
    
    def generate_dataset(self, n_samples: int, error_rate: Optional[float] = None) -> Dict:
        """
        Generate dataset of error-syndrome pairs
        
        Args:
            n_samples: Number of samples to generate
            error_rate: Error rate (if None, use noise model default)
            
        Returns:
            Dictionary with stacked arrays
        """
        samples = [self.generate_error_syndrome_pair(error_rate) for _ in range(n_samples)]
        
        return {
            'error_X': np.stack([s['error_X'] for s in samples]),
            'error_Z': np.stack([s['error_Z'] for s in samples]),
            'syndrome_X': np.stack([s['syndrome_X'] for s in samples]),
            'syndrome_Z': np.stack([s['syndrome_Z'] for s in samples]),
            'error_rate': samples[0]['error_rate']
        }
    
    def apply_measurement_errors(self, syndrome: np.ndarray, measurement_error_rate: float = 0.01) -> np.ndarray:
        """
        Add measurement errors to syndrome
        
        Args:
            syndrome: Perfect syndrome
            measurement_error_rate: Probability of bit flip in syndrome
            
        Returns:
            Noisy syndrome
        """
        noisy_syndrome = syndrome.copy()
        measurement_errors = np.random.random(syndrome.shape) < measurement_error_rate
        noisy_syndrome = (noisy_syndrome + measurement_errors) % 2
        return noisy_syndrome.astype(np.int8)


class BiasedNoiseSimulator(ErrorSimulator):
    """Simulator with biased noise (e.g., more Z errors than X errors)"""
    
    def __init__(self, H_X: np.ndarray, H_Z: np.ndarray, bias: float = 10.0, base_rate: float = 0.01):
        """
        Initialize biased noise simulator
        
        Args:
            H_X: X-type stabilizer matrix
            H_Z: Z-type stabilizer matrix
            bias: Ratio of Z:X error rates (bias > 1 means more Z errors)
            base_rate: Base error probability
        """
        # Calculate p_x and p_z such that p_z/p_x = bias and p_x + p_z ≈ base_rate
        p_x = base_rate / (1 + bias)
        p_z = bias * p_x
        
        noise_model = NoiseModel(p_x=p_x, p_y=0, p_z=p_z)
        super().__init__(H_X, H_Z, noise_model)
        self.bias = bias


def compute_logical_error(error: np.ndarray, logical_ops: np.ndarray) -> bool:
    """
    Check if error causes logical failure
    
    Args:
        error: Error vector
        logical_ops: Logical operator matrix (k × n)
        
    Returns:
        True if any logical operator anticommutes with error
    """
    # Logical error if odd overlap with any logical operator
    overlaps = (logical_ops @ error) % 2
    return np.any(overlaps == 1)


if __name__ == "__main__":
    # Test with toric code
    from qldpc_codes import ToricCode
    
    print("=== Testing Error Simulator ===")
    toric = ToricCode(L=4)
    simulator = ErrorSimulator(toric.H_X, toric.H_Z, NoiseModel(p_depol=0.05))
    
    # Generate single error-syndrome pair
    print("\n--- Single Sample ---")
    sample = simulator.generate_error_syndrome_pair()
    print(f"Error X weight: {sample['error_X'].sum()}")
    print(f"Error Z weight: {sample['error_Z'].sum()}")
    print(f"Syndrome X weight: {sample['syndrome_X'].sum()}")
    print(f"Syndrome Z weight: {sample['syndrome_Z'].sum()}")
    
    # Verify syndrome calculation
    syndrome_X_check = (toric.H_X @ sample['error_Z']) % 2
    syndrome_Z_check = (toric.H_Z @ sample['error_X']) % 2
    print(f"Syndrome X correct: {np.array_equal(syndrome_X_check, sample['syndrome_X'])}")
    print(f"Syndrome Z correct: {np.array_equal(syndrome_Z_check, sample['syndrome_Z'])}")
    
    # Generate dataset
    print("\n--- Dataset Generation ---")
    dataset = simulator.generate_dataset(n_samples=100)
    print(f"Dataset shapes:")
    print(f"  error_X: {dataset['error_X'].shape}")
    print(f"  error_Z: {dataset['error_Z'].shape}")
    print(f"  syndrome_X: {dataset['syndrome_X'].shape}")
    print(f"  syndrome_Z: {dataset['syndrome_Z'].shape}")
    
    # Statistics
    print(f"\nAverage error weight: {dataset['error_X'].sum(axis=1).mean():.2f}")
    print(f"Average syndrome weight: {dataset['syndrome_X'].sum(axis=1).mean():.2f}")
    
    # Test biased noise
    print("\n=== Biased Noise (Z-bias = 10) ===")
    biased_sim = BiasedNoiseSimulator(toric.H_X, toric.H_Z, bias=10.0, base_rate=0.05)
    biased_dataset = biased_sim.generate_dataset(n_samples=100)
    
    avg_x_errors = biased_dataset['error_X'].sum(axis=1).mean()
    avg_z_errors = biased_dataset['error_Z'].sum(axis=1).mean()
    print(f"Average X errors: {avg_x_errors:.2f}")
    print(f"Average Z errors: {avg_z_errors:.2f}")
    print(f"Measured bias (Z/X): {avg_z_errors / max(avg_x_errors, 0.1):.2f}")
    
    # Test measurement errors
    print("\n=== Measurement Errors ===")
    perfect_syndrome = sample['syndrome_X']
    noisy_syndrome = simulator.apply_measurement_errors(perfect_syndrome, measurement_error_rate=0.1)
    bit_flips = (perfect_syndrome != noisy_syndrome).sum()
    print(f"Syndrome length: {len(perfect_syndrome)}")
    print(f"Measurement bit flips: {bit_flips}")
