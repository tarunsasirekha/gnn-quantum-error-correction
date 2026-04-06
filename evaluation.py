"""
Evaluation and Benchmarking Module
Compare GNN decoders against classical baselines
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class DecoderResult:
    """Results from decoder evaluation"""
    name: str
    success_rate: float
    logical_error_rate: float
    avg_decode_time: float
    avg_error_weight: float
    avg_residual_weight: float
    frame_error_rate: float


class DecoderEvaluator:
    """Evaluate and compare decoders"""
    
    def __init__(self, H_X: np.ndarray, H_Z: np.ndarray, logical_ops_X: Optional[np.ndarray] = None):
        """
        Initialize evaluator
        
        Args:
            H_X: X-type stabilizer matrix
            H_Z: Z-type stabilizer matrix
            logical_ops_X: Logical X operator matrix [k, n]
        """
        self.H_X = H_X
        self.H_Z = H_Z
        self.logical_ops_X = logical_ops_X
        self.n_qubits = H_X.shape[1]
    
    def evaluate_classical_decoder(self,
                                   decoder,
                                   syndromes: np.ndarray,
                                   true_errors: np.ndarray,
                                   name: str = "Classical") -> DecoderResult:
        """
        Evaluate classical decoder
        
        Args:
            decoder: Decoder with .decode() method
            syndromes: Test syndromes [n_samples, n_checks]
            true_errors: True errors [n_samples, n_qubits]
            name: Decoder name
            
        Returns:
            DecoderResult
        """
        n_samples = syndromes.shape[0]
        
        successes = 0
        logical_failures = 0
        total_time = 0
        total_residual_weight = 0
        
        for i in range(n_samples):
            # Time decoding
            start_time = time.time()
            decoded_error = decoder.decode(syndromes[i])
            decode_time = time.time() - start_time
            total_time += decode_time
            
            # Check syndrome matching
            decoded_syndrome = (self.H_Z @ decoded_error) % 2
            syndrome_match = np.array_equal(decoded_syndrome, syndromes[i])
            
            if syndrome_match:
                successes += 1
            
            # Check logical error
            residual = (true_errors[i] + decoded_error) % 2
            total_residual_weight += residual.sum()
            
            if self.logical_ops_X is not None:
                overlaps = (self.logical_ops_X @ residual) % 2
                if np.any(overlaps == 1):
                    logical_failures += 1
        
        avg_error_weight = true_errors.sum(axis=1).mean()
        
        return DecoderResult(
            name=name,
            success_rate=successes / n_samples,
            logical_error_rate=logical_failures / n_samples if self.logical_ops_X is not None else 0.0,
            avg_decode_time=total_time / n_samples,
            avg_error_weight=avg_error_weight,
            avg_residual_weight=total_residual_weight / n_samples,
            frame_error_rate=1 - (successes / n_samples)
        )
    
    def evaluate_gnn_decoder(self,
                            model: torch.nn.Module,
                            graph_data_list: List,
                            true_errors: np.ndarray,
                            device: str = 'cpu',
                            name: str = "GNN") -> DecoderResult:
        """
        Evaluate GNN decoder
        
        Args:
            model: GNN model
            graph_data_list: List of graph Data objects
            true_errors: True errors [n_samples, n_qubits]
            device: Device to run on
            name: Decoder name
            
        Returns:
            DecoderResult
        """
        model.eval()
        n_samples = len(graph_data_list)
        
        successes = 0
        logical_failures = 0
        total_time = 0
        total_residual_weight = 0
        
        with torch.no_grad():
            for i, data in enumerate(graph_data_list):
                data = data.to(device)
                
                # Time decoding
                start_time = time.time()
                predictions = model(data)
                pred_probs = torch.sigmoid(predictions.squeeze())
                decoded_error = (pred_probs > 0.5).cpu().numpy().astype(np.int8)
                decode_time = time.time() - start_time
                total_time += decode_time
                
                # Check syndrome matching
                decoded_syndrome = (self.H_Z @ decoded_error) % 2
                syndrome_match = np.array_equal(decoded_syndrome, data.syndrome.cpu().numpy())
                
                if syndrome_match:
                    successes += 1
                
                # Check logical error
                residual = (true_errors[i] + decoded_error) % 2
                total_residual_weight += residual.sum()
                
                if self.logical_ops_X is not None:
                    overlaps = (self.logical_ops_X @ residual) % 2
                    if np.any(overlaps == 1):
                        logical_failures += 1
        
        avg_error_weight = true_errors.sum(axis=1).mean()
        
        return DecoderResult(
            name=name,
            success_rate=successes / n_samples,
            logical_error_rate=logical_failures / n_samples if self.logical_ops_X is not None else 0.0,
            avg_decode_time=total_time / n_samples,
            avg_error_weight=avg_error_weight,
            avg_residual_weight=total_residual_weight / n_samples,
            frame_error_rate=1 - (successes / n_samples)
        )
    
    def compare_decoders(self, results: List[DecoderResult]) -> None:
        """Print comparison table"""
        print("\n" + "="*80)
        print("DECODER COMPARISON")
        print("="*80)
        print(f"{'Decoder':<20} {'Success%':<12} {'Logical Err%':<15} {'Time (ms)':<12} {'Residual Wt':<12}")
        print("-"*80)
        
        for result in results:
            print(f"{result.name:<20} "
                  f"{result.success_rate*100:>10.2f}%  "
                  f"{result.logical_error_rate*100:>13.2f}%  "
                  f"{result.avg_decode_time*1000:>10.3f}  "
                  f"{result.avg_residual_weight:>10.2f}")
        
        print("="*80)
    
    def plot_results(self, results: List[DecoderResult], save_path: Optional[str] = None):
        """Plot comparison charts"""
        names = [r.name for r in results]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Success rate
        axes[0, 0].bar(names, [r.success_rate * 100 for r in results])
        axes[0, 0].set_ylabel('Success Rate (%)')
        axes[0, 0].set_title('Decoding Success Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Logical error rate
        if results[0].logical_error_rate > 0:
            axes[0, 1].bar(names, [r.logical_error_rate * 100 for r in results])
            axes[0, 1].set_ylabel('Logical Error Rate (%)')
            axes[0, 1].set_title('Logical Error Rate')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Decode time
        axes[1, 0].bar(names, [r.avg_decode_time * 1000 for r in results])
        axes[1, 0].set_ylabel('Time (ms)')
        axes[1, 0].set_title('Average Decode Time')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Residual weight
        axes[1, 1].bar(names, [r.avg_residual_weight for r in results])
        axes[1, 1].set_ylabel('Residual Error Weight')
        axes[1, 1].set_title('Average Residual Error Weight')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class ScalabilityTester:
    """Test decoder scalability with increasing code size"""
    
    def __init__(self):
        self.results = []
    
    def test_code_size_scaling(self,
                              code_sizes: List[int],
                              decoder_factory,
                              error_rate: float = 0.01,
                              n_trials: int = 100) -> Dict:
        """
        Test how decoder scales with code size
        
        Args:
            code_sizes: List of code sizes (e.g., toric code L values)
            decoder_factory: Function that creates decoder for given size
            error_rate: Physical error rate
            n_trials: Number of trials per size
            
        Returns:
            Dictionary of results
        """
        from qldpc_codes import ToricCode
        from error_simulation import ErrorSimulator, NoiseModel
        
        results = {
            'sizes': [],
            'n_qubits': [],
            'success_rates': [],
            'decode_times': [],
            'memory_usage': []
        }
        
        for L in code_sizes:
            print(f"\nTesting L={L}...")
            
            # Create code
            toric = ToricCode(L)
            simulator = ErrorSimulator(toric.H_X, toric.H_Z, NoiseModel(p_depol=error_rate))
            
            # Generate test data
            dataset = simulator.generate_dataset(n_trials)
            
            # Create decoder
            decoder = decoder_factory(toric)
            
            # Evaluate
            successes = 0
            total_time = 0
            
            for i in range(n_trials):
                start = time.time()
                decoded = decoder.decode(dataset['syndrome_Z'][i])
                total_time += time.time() - start
                
                decoded_syndrome = (toric.H_Z @ decoded) % 2
                if np.array_equal(decoded_syndrome, dataset['syndrome_Z'][i]):
                    successes += 1
            
            results['sizes'].append(L)
            results['n_qubits'].append(toric.n_qubits)
            results['success_rates'].append(successes / n_trials)
            results['decode_times'].append(total_time / n_trials)
            
            print(f"  Qubits: {toric.n_qubits}")
            print(f"  Success rate: {successes/n_trials:.2%}")
            print(f"  Avg decode time: {total_time/n_trials*1000:.2f} ms")
        
        self.results.append(results)
        return results
    
    def plot_scaling(self, results: Dict, title: str = "Decoder Scaling"):
        """Plot scaling results"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Success rate vs size
        axes[0].plot(results['n_qubits'], np.array(results['success_rates']) * 100, 'o-')
        axes[0].set_xlabel('Number of Qubits')
        axes[0].set_ylabel('Success Rate (%)')
        axes[0].set_title(f'{title}: Success Rate')
        axes[0].grid(True)
        
        # Decode time vs size
        axes[1].plot(results['n_qubits'], np.array(results['decode_times']) * 1000, 'o-')
        axes[1].set_xlabel('Number of Qubits')
        axes[1].set_ylabel('Decode Time (ms)')
        axes[1].set_title(f'{title}: Decode Time')
        axes[1].grid(True)
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        plt.show()


def threshold_analysis(code_factory,
                      decoder_factory,
                      error_rates: List[float],
                      n_trials: int = 1000) -> Dict:
    """
    Perform threshold analysis: find error rate where logical error rate = physical error rate
    
    Args:
        code_factory: Function to create code
        decoder_factory: Function to create decoder
        error_rates: List of physical error rates to test
        n_trials: Trials per rate
        
    Returns:
        Results dictionary
    """
    from error_simulation import ErrorSimulator, NoiseModel
    
    results = {
        'error_rates': error_rates,
        'logical_error_rates': [],
        'frame_error_rates': []
    }
    
    code = code_factory()
    decoder = decoder_factory(code)
    logical_ops_X, _ = code.get_logical_operators()
    
    for p in error_rates:
        print(f"\nTesting error rate p={p:.4f}...")
        
        simulator = ErrorSimulator(code.H_X, code.H_Z, NoiseModel(p_depol=p))
        
        logical_failures = 0
        frame_errors = 0
        
        for _ in range(n_trials):
            sample = simulator.generate_error_syndrome_pair()
            
            # Decode
            decoded = decoder.decode(sample['syndrome_Z'])
            
            # Check frame error (syndrome mismatch)
            decoded_syndrome = (code.H_Z @ decoded) % 2
            if not np.array_equal(decoded_syndrome, sample['syndrome_Z']):
                frame_errors += 1
            
            # Check logical error
            residual = (sample['error_X'] + decoded) % 2
            overlaps = (logical_ops_X @ residual) % 2
            if np.any(overlaps == 1):
                logical_failures += 1
        
        logical_error_rate = logical_failures / n_trials
        frame_error_rate = frame_errors / n_trials
        
        results['logical_error_rates'].append(logical_error_rate)
        results['frame_error_rates'].append(frame_error_rate)
        
        print(f"  Logical error rate: {logical_error_rate:.4f}")
        print(f"  Frame error rate: {frame_error_rate:.4f}")
    
    return results


if __name__ == "__main__":
    from qldpc_codes import ToricCode
    from error_simulation import ErrorSimulator, NoiseModel
    from classical_decoders import BeliefPropagationDecoder, GreedyDecoder
    from graph_representation import BatchGraphBuilder
    from gnn_models import GNNDecoder
    import torch
    
    print("=== Testing Evaluation Module ===")
    
    # Create code
    toric = ToricCode(L=4)
    logical_ops_X, _ = toric.get_logical_operators()
    
    # Generate test data
    simulator = ErrorSimulator(toric.H_X, toric.H_Z, NoiseModel(p_depol=0.05))
    test_dataset = simulator.generate_dataset(n_samples=50)
    
    # Create evaluator
    evaluator = DecoderEvaluator(toric.H_X, toric.H_Z, logical_ops_X)
    
    # Test classical decoders
    print("\n--- Evaluating Classical Decoders ---")
    bp_decoder = BeliefPropagationDecoder(toric.H_Z, max_iterations=50)
    bp_result = evaluator.evaluate_classical_decoder(
        bp_decoder,
        test_dataset['syndrome_Z'],
        test_dataset['error_X'],
        name="Belief Propagation"
    )
    
    greedy_decoder = GreedyDecoder(toric.H_Z)
    greedy_result = evaluator.evaluate_classical_decoder(
        greedy_decoder,
        test_dataset['syndrome_Z'],
        test_dataset['error_X'],
        name="Greedy"
    )
    
    # Compare
    evaluator.compare_decoders([bp_result, greedy_result])
    
    print("\n=== Evaluation module tests passed! ===")
